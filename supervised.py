import argparse
import logging
import os
import pprint
import time
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch.distributed as dist
import numpy as np
import random
# from evaluate import evaluate
from dataset.semi_crop import SemiDataset
from dataset.val import ValDataset
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, init_log, intersectionAndUnion
from util.dist_helper import setup_distributed
from model.semseg.deeplabv3plus import DeepLabV3Plus
import torch.nn.functional as F
# from model.semseg.models_samrs import SemsegFinetuneFramework

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
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


# @torch.no_grad()
@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    predict_meter = AverageMeter()

    model.eval()

    for (x, y) in valid_loader:
        x = x.cuda()
        if cfg['eval_mode'] == 'slide_window':
            b, _, h, w = x.shape    # 获取输入图像的尺寸 (batch, channels, height, width)
            final = torch.zeros(b, cfg['nclass'], h, w).cuda()  # 用于存储最终预测结果
            size = cfg['crop_size']
            step = 510
            b = 0
            a = 0
            while (a <= int(h / step)):
                while (b <= int(w / step)):
                    sub_input = x[:,:, min(a * step, h - size): min(a * step + size, h), min(b * step, w - size):min(b * step + size, w)]
                    # print("sub_input.shape", sub_input.shape)
                    mask = model(sub_input) 
                    final[:,:, min(a * step, h - size): min(a * step + size, h), min(b * step, w - size):min(b * step + size, w)] += mask
                    b += 1
                b = 0
                a += 1
            o = final.argmax(dim=1)
        
        elif cfg['eval_mode'] == 'resize':
        # 使用缩放方式进行预测
            original_shape = x.shape[-2:]  # 保存原始图像的尺寸 (h, w)
            resized_x = F.interpolate(x, size=cfg['crop_size'], mode='bilinear', align_corners=True)
            resized_o = model(resized_x)   
            # 将预测结果复原到原始尺寸
            o = F.interpolate(resized_o, size=original_shape, mode='bilinear', align_corners=True)
            o = o.argmax(dim=1)

        else:
            # 直接进行预测（非滑动窗口模式）
            o = model(x)
            o = o.max(1)[1]
        gray = np.uint8(o.cpu().numpy())
        target = np.array(y, dtype=np.int32)
        intersection, union, target, predict = intersectionAndUnion(gray, target, cfg['nclass'], cfg['ignore_index'])
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        predict_meter.update(predict)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    precise_class = intersection_meter.sum / (predict_meter.sum + 1e-10)
    F1_class = 2*(precise_class*accuracy_class) / (precise_class+accuracy_class)
    
    mIoU = np.nanmean(iou_class) * 100.0
    mAcc = np.nanmean(accuracy_class) * 100.0
    mF1 = np.nanmean(F1_class) * 100.0
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    return mIoU, mAcc, mF1, allAcc, iou_class, F1_class





def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

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

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    trainset = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', size=cfg['crop_size'], ignore_value=cfg['ignore_index'], id_path=args.labeled_id_path)
    valset = ValDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size,rank=rank)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=(trainsampler is None),
                             pin_memory=True, num_workers=8, drop_last=False, sampler=trainsampler)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset, num_replicas=world_size, rank=rank)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=8,
                           drop_last=False, sampler=valsampler)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    scaler = torch.cuda.amp.GradScaler()
    amp = cfg['amp']

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        
        total_loss = AverageMeter()
        trainsampler.set_epoch(epoch)

        for i, (img, mask) in enumerate(trainloader):
            img, mask = img.cuda(), mask.cuda()

            with torch.cuda.amp.autocast(enabled=amp):
                model.train()
                pred = model(img)
                sup_loss = criterion(pred, mask)                
                torch.distributed.barrier()
                optimizer.zero_grad()
                loss = scaler.scale(sup_loss)
                loss.backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss.update(sup_loss)
            iters = epoch * len(trainloader) + i

            if rank == 0:
                writer.add_scalar('train/loss_all', sup_loss.item(), iters)
                writer.add_scalar('train/loss_x', sup_loss.item(), iters)
            
            if (i % (max(2, len(trainloader) // 8)) == 0) and (rank == 0):
                # logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, sup_loss.item()))

        scheduler.step()
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


if __name__ == '__main__':
    set_seeds(1234)
    main()
