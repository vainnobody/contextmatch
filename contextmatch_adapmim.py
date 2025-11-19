import os
import time
import yaml
import pprint
import logging
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from supervised import validation_cpu
from dataset.semi_crop import SemiDataset
from dataset.val import ValDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter, intersectionAndUnionGPU
from util.dist_helper import setup_distributed
import torch.nn.functional as F
from util.train_utils import (DictAverageMeter, confidence_weighted_loss,
                               cutmix_img_, cutmix_mask)
import torch.distributed as dist
from util.dist_helper import setup_distributed

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--interval', default=1, type=int, help='valid interval')




def set_seeds(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_weighted_iou(pre, mask, cfg):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    # 计算交集、并集和目标类别数量
    intersection, union, target, predict = intersectionAndUnionGPU(pre, mask, cfg['nclass'], cfg['ignore_index'])
    dist.all_reduce(intersection)
    dist.all_reduce(union)
    dist.all_reduce(target)

    # 更新计量器
    intersection_meter.update(intersection.cpu().numpy())
    union_meter.update(union.cpu().numpy())
    target_meter.update(target.cpu().numpy())

    # 计算每个类别的IoU
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)

    # 计算每个类别的权重（根据目标类别数量）
    class_weights = target_meter.sum / (target_meter.sum.sum() + 1e-10)

    # 计算加权IoU
    weighted_iou = np.sum(iou_class * class_weights)

    return weighted_iou


def compute_entropy(prob):
    # 计算给定概率分布的熵值
    log_prob = torch.log(prob + 1e-10)  # 加一个小常数防止log(0)
    entropy = -torch.sum(prob * log_prob, dim=1)  # 沿类别轴求和
    return entropy

def compute_uncertainty(pred_u_w, pred_u_s):
    # 计算两个预测的平均熵值作为不确定性度量
    entropy_w = compute_entropy(F.softmax(pred_u_w, dim=1))
    entropy_s = compute_entropy(F.softmax(pred_u_s, dim=1))
    uncertainty_map = (entropy_w + entropy_s) / 2.0
    return uncertainty_map

def compute_mask_ratio(avg_uncertainty, iou):
    B, _, _ = avg_uncertainty.shape
    # 初始化mask_ratios
    mask_ratios = torch.zeros_like(avg_uncertainty)

    normalized_uncertainty = (avg_uncertainty - avg_uncertainty.min()) / \
                                 (avg_uncertainty.max() - avg_uncertainty.min()) 
    mask_ratios = iou * (1 - normalized_uncertainty)

    return mask_ratios

def compute_patch_uncertainty(uncertainty_map, patch_size=64):
    B, H, W = uncertainty_map.shape
    L = H // patch_size
    # 将不确定性图转换为patch形式
    patches = uncertainty_map.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(B, L, L, patch_size * patch_size)
    # 计算每个patch的平均不确定性
    avg_uncertainty = patches.mean(dim=-1)
    return avg_uncertainty

def compute_symmetric_iou(batch_pred_w, batch_pred_s, cfg):
    # 计算对称IoU
    # batch_iou = []
    iou_w_s = calculate_weighted_iou(batch_pred_w.argmax(dim=1), batch_pred_s.argmax(dim=1), cfg)
    iou_s_w = calculate_weighted_iou(batch_pred_s.argmax(dim=1), batch_pred_w.argmax(dim=1), cfg)
    symmetric_iou = (iou_w_s + iou_s_w) / 2.0
        # batch_iou.append(symmetric_iou)
    return symmetric_iou




def patch_mask_adaptive(img_tensor, uncertainty_map, patch_size=16, mask_ratio=0.5, p=0.5):
    """
    对输入的批量图像张量进行 patch 级别的掩码，根据不确定性动态掩码

    Parameters:
    img_tensor (torch.Tensor): 输入的图像张量，假设其形状为 [B, C, H, W]
    uncertainty_map (torch.Tensor): 归一化的不确定性矩阵，形状应为 [B, 1, H // patch_size, W // patch_size]
    patch_size (int): 每个 patch 的尺寸
    mask_ratio (float): 需要被掩码的 patch 比例

    Returns:
    torch.Tensor: 掩码后的图像张量
    """
    assert img_tensor.dim() == 4, "Input tensor should be 4D (B, C, H, W)"
    assert uncertainty_map.dim() == 4, "Uncertainty map should be 4D (B, 1, H // patch_size, W // patch_size)"
    
    B, C, H, W = img_tensor.shape
    device = img_tensor.device  # 获取 tensor 所在的设备，确保 mask 在同一设备
    # 初始化一个全为1的 mask 张量
    mask = torch.ones((B, C, H, W), device=device)

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    total_patches = num_patches_h * num_patches_w
    num_mask_patches = int(total_patches * mask_ratio)

    # 对批量中的每张图像应用掩码
    for i in range(B):
        # 从不确定性图中获取该图像的 patch 不确定性
        uncertainty_scores = uncertainty_map[i, 0]  # 获取每张图像的646不确定性分数

        # Flatten the uncertainty scores for sorting
        uncertainty_scores_flat = uncertainty_scores.view(-1)

        # 根据不确定性分数排序，优先选取低分数的 patch
        sorted_indices = torch.argsort(uncertainty_scores_flat)

        # 随机选择一些 patch 进行掩码，但优先选择不确定性低的 patch
        mask_indices = sorted_indices[:num_mask_patches]

        for index in mask_indices:
            row = index // num_patches_w
            col = index % num_patches_w

            row_start = row * patch_size
            col_start = col * patch_size
            mask[i, :, row_start:row_start + patch_size, col_start:col_start + patch_size] = 0

    # 选择不确定性高的区域mask
    masked_img = img_tensor * (1 - mask)

    return masked_img



@torch.no_grad()
def validation(args, model, valid_loader, cfg):

    intersection_meter = AverageMeter()
    union_meter  = AverageMeter()
    target_meter  = AverageMeter()
    predict_meter = AverageMeter()

    model.eval()

    for (x, y) in valid_loader:

        x, y = x.cuda(), y.long().cuda()

        o = model.forward(x)

        o = o.max(1)[1]
        intersection, union, target, predict = intersectionAndUnionGPU(o, y, cfg['nclass'], cfg['ignore_index'])
        # if args.distributed=='True':
        dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target), dist.all_reduce(predict)
        intersection, union, target, predict = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), predict.cpu().numpy(),
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target), predict_meter.update(predict)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10) * 100
    precise_class = intersection_meter.sum / (predict_meter.sum + 1e-10) * 100
    F1_class = 2*(precise_class*accuracy_class) / (precise_class+accuracy_class) * 100

    if cfg['dataset'] == 'isaid_ori':
        mIoU = np.mean(iou_class[1:])
        mAcc = np.mean(accuracy_class[1:])
        mF1 = np.mean(F1_class[1:])
        allAcc = sum(intersection_meter.sum[1:]) / (sum(target_meter.sum[1:]) + 1e-10)
    else:
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        mF1 = np.mean(F1_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    return mIoU, mAcc, mF1, allAcc, iou_class


class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def to_image(cmap, mask):
    # print(cmap, mask.shape)
    cmap_tensor = torch.tensor(cmap, dtype=torch.uint8)
    mask = mask.squeeze(0)

    # 为mask中的每个值找到对应颜色
    # 注意：由于直接索引可能导致性能问题，考虑到分段操作或其他优化
    rgb_mask = cmap_tensor[mask]

    return rgb_mask.permute(2, 0, 1)  # 调整维度以符合C*H*W的格式

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    # Save the best checkpoint if needed
    if is_best:
        best_filename = filename.replace('latest', 'best')
        torch.save(state, best_filename)

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)
    ddp = True if world_size > 1 else False
    amp = cfg['amp']

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    conf_thresh = cfg['conf_thresh']

    model = DeepLabV3Plus(cfg)
    from mmengine.optim import build_optim_wrapper
    optim_wrapper = dict(
        optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01),
        paramwise_cfg=dict(
            custom_keys={
                'absolute_pos_embed': dict(decay_mult=0.),
                'relative_position_bias_table': dict(decay_mult=0.),
                'norm': dict(decay_mult=0.)
            }))
    
    
    optimizer = build_optim_wrapper(model, optim_wrapper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, cfg['epochs'], eta_min=0, last_epoch=-1)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    if ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                        output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             size=cfg['crop_size'], ignore_value=cfg['ignore_index'], id_path=args.unlabeled_id_path)
    
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             size=cfg['crop_size'], ignore_value=cfg['ignore_index'], id_path=args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = ValDataset(cfg['dataset'], cfg['data_root'], 'val')

    if ddp:
        trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
        trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                                   pin_memory=True, num_workers=8, drop_last=False, sampler=trainsampler_l)
        trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
        trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                                   pin_memory=True, num_workers=8, drop_last=False, sampler=trainsampler_u)
        valsampler = torch.utils.data.distributed.DistributedSampler(valset)
        valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=8,
                               drop_last=False, sampler=valsampler)
    else:
        trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                                pin_memory=True, num_workers=4, shuffle=True, drop_last=True)
        trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                                pin_memory=True, num_workers=1, shuffle=True, drop_last=True)
        valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    total_epochs = cfg['epochs']
    total_iters = len(trainloader_u) * total_epochs
    epoch = -1
    previous_best = 0.0
    ETA = 0.0
    scaler = torch.cuda.amp.GradScaler()
    is_best = False

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    

    for epoch in range(epoch + 1, cfg['epochs']):
        log_avg = DictAverageMeter()
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        if ddp:
            trainloader_l.sampler.set_epoch(epoch)
            trainloader_u.sampler.set_epoch(epoch)


        start_time = time.time()
        model.train()
        total_loss = AverageMeter()
        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2, _),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _, _)) in enumerate(loader):
            
            
            t0 = time.time()
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()
            iters = epoch * len(trainloader_u) + i

            # CutMix images
            cutmix_img_(img_u_s1, img_u_s1_mix, cutmix_box1)

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
            with torch.no_grad():
                model.eval()
                pred_u_w, pred_u_s = model(torch.cat((img_u_w, img_u_s1))).split([num_lb, num_ulb])
                uncertainty_maps = compute_uncertainty(pred_u_w, pred_u_s)
                patch_uncertainty_maps = compute_patch_uncertainty(uncertainty_maps, patch_size=cfg['PATCH_SIZE'])
                wIoU = compute_symmetric_iou(pred_u_w, pred_u_s, cfg)
                mask_ratios = compute_mask_ratio(patch_uncertainty_maps, wIoU)
                MIM_Ratio =  mask_ratios.mean() * cfg['MAX_RATIO']

                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix, mask_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)

            img_u_w_masked = patch_mask_adaptive(img_u_w, patch_uncertainty_maps.unsqueeze(1), cfg['PATCH_SIZE'], MIM_Ratio)
            
            with torch.cuda.amp.autocast(enabled=amp):
                model.train()
                
                preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
                pred_x, pred_u_w = preds.split([num_lb, num_ulb])
                pred_u_w_fp = preds_fp[num_lb:]

                pred_u_s1, pred_u_w_masked = model(torch.cat((img_u_s1, img_u_w_masked))).split([num_ulb, num_ulb])
                pred_u_w = pred_u_w.detach()
                conf_u_w, mask_u_w = pred_u_w.softmax(dim=1).max(dim=1)

                mask_u_w_cutmixed1 = cutmix_mask(mask_u_w, mask_u_w_mix, cutmix_box1)
                conf_u_w_cutmixed1 = cutmix_mask(conf_u_w, conf_u_w_mix, cutmix_box1)
                ignore_mask_cutmixed1 = cutmix_mask(ignore_mask, ignore_mask_mix, cutmix_box1)
                           
                loss_x = criterion_l(pred_x, mask_x)
                loss_u_s = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
                loss_u_s = confidence_weighted_loss(loss_u_s, conf_u_w_cutmixed1, ignore_mask_cutmixed1, cfg['ignore_index'], conf_thresh=conf_thresh)
                
                loss_u_w_masked = criterion_u(pred_u_w_masked, mask_u_w)
                loss_u_w_masked = confidence_weighted_loss(loss_u_w_masked, conf_u_w, ignore_mask, cfg['ignore_index'], conf_thresh=conf_thresh)

                loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
                loss_u_w_fp = confidence_weighted_loss(loss_u_w_fp, conf_u_w, ignore_mask,  cfg['ignore_index'], conf_thresh=conf_thresh)

                mask_ratio = ((conf_u_w >= conf_thresh) & (ignore_mask != cfg['ignore_index'])).sum().item() / (ignore_mask != cfg['ignore_index']).sum()
                loss_standard = loss_u_s * 0.25 + loss_u_w_masked * 0.25 + loss_u_w_fp * 0.5
                total_loss = (loss_x + loss_standard) / 2.0
 
            if ddp:
                torch.distributed.barrier()

            optimizer.zero_grad()
            if amp:
                loss = scaler.scale(total_loss)
                # torch.autograd.set_detect_anomaly(True)
                loss.backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            # Logging
            log_avg.update({
                'iter time': time.time() - t0,
                'Total loss': total_loss,          # Logging original loss (=total loss), not AMP scaled loss
                'Loss x': loss_x,
                'Loss u_s': loss_u_s,
                'loss_u_w_masked': loss_u_w_masked,
                'Mask ratio': mask_ratio,
            })


            if (i % (max(2, len(trainloader_u) // 8)) == 0) and (rank == 0):
                # logger.info('Training epoch [{}/{}] iter [{}/{}]: loss {:.4f}'.format(epoch+1, args.epochs, i+1, len(train_loader_u), total_loss.avg))
                logger.info('===========> Iteration: {:}/{:}, Epoch: {:}/{:}, LR: {:.5f}, MIM_Ratio: {:.2f}, log_avg: {}'
                    .format(i+1, len(trainloader_u), epoch+1, cfg['epochs'], optimizer.param_groups[0]['lr'], MIM_Ratio, str(log_avg)))
     

        if epoch % args.interval == 0:
            start_time = time.time()
            mIoU, mAcc, mF1, allAcc, iou_class, F1_class = validation_cpu(cfg, model, valloader)
            end_time = time.time()

            if rank == 0:
                for (cls_idx, iou) in enumerate(iou_class):
                    logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                                'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
                # logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
                logger.info('Last: validation epoch [{}/{}]: mIoU/mAcc/mF1/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}. Cost {:.4f} secs'.format(epoch+1, cfg['epochs'], mIoU, mAcc, mF1, allAcc, end_time-start_time))
                    
                writer.add_scalar('eval/mIoU', mIoU, epoch)
                for i, iou in enumerate(iou_class):
                    writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

            is_best = mIoU > previous_best
            previous_best = max(mIoU, previous_best)
            if rank == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'previous_best': previous_best,
                }
                torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
                if is_best:
                    torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
        scheduler.step()

if __name__ == '__main__':
    set_seeds(1234)
    main()
