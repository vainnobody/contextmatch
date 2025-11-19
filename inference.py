import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import yaml
import random
from dataset.semi import SemiDataset
# from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.deeplabv3plus_scalematch import DeepLabV3Plus
# from model.semseg.deeplabv3plus_mscale_fp_convfusion import DeepLabV3Plus
# from model.semseg.deeplabv3plus_mscale_featureConsistency_fp import DeepLabV3Plus
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log, color_map
from util.dist_helper import setup_distributed
from collections import OrderedDict
from PIL import Image
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--ckpt-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)






def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    # if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_rgb(cmap, mask):
    # print(cmap, mask.shape)
    cmap_tensor = torch.tensor(cmap, dtype=torch.uint8)
    mask = mask.squeeze(0)

    # 为mask中的每个值找到对应颜色
    # 注意：由于直接索引可能导致性能问题，考虑到分段操作或其他优化
    rgb_mask = cmap_tensor[mask]

    return np.array(rgb_mask)  # 调整维度以符合C*H*W的格式


def evaluate(model, loader, mode, cfg, eval_scales=None):

    model.eval()
    # assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    cmap = color_map(dataset=args.dataset)

    with torch.no_grad():
        for img, mask, id in loader:
            
            img = img.cuda()
            b, _, h, w = img.shape

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        sub_img = img[:, :, row: min(h, row + grid), col: min(w, col + grid)]
                        # ori_size = sub_img.size(2), sub_img.size(3)
                        # sub_img = F.interpolate(sub_img, scale_factor=random_size, mode='bilinear', align_corners=True)
                        pred = model(sub_img, scales=eval_scales,eval_mode='atten_fusion')
                        # pred = F.interpolate(pred, size=ori_size, mode='bilinear', align_corners=True)
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)
                # rgb_pre = to_rgb(cmap, pred.cpu())
                # result = Image.fromarray(rgb_pre)
                # filename = id[0].split('/')[-1]
                # result.save('/data/users/zhengzhiyu/SSL/SSSS_LL/UniMatch-main/prediction_result/cityscapes/'
                #             '1_8/fixmatch/scale_2.0/' + filename)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                if eval_scales != None and mode == 'add':
                    pred = model(img, scales=eval_scales, fusion_mode='add').argmax(dim=1)
                    rgb_pre = to_rgb(cmap, pred.cpu())
                    # result = Image.fromarray(rgb_pre)
                    # result.save('/data/users/zhengzhiyu/SSL/SSSS_LL/UniMatch-main/prediction_result/1_16/scalematch_pascal_1_16_add_fusion/' + id[0][-15:])

                elif eval_scales != None and mode == 'atten_fusion':
                    pred = model(img, scales=eval_scales, eval_mode='atten_fusion').argmax(dim=1)
                    # rgb_pre = to_rgb(cmap, pred.cpu())
                    # result = Image.fromarray(rgb_pre)
                    # result.save( '/data/users/zhengzhiyu/SSL/SSSS_LL/UniMatch-main/prediction_result/1_16/scalematch_pascal_1_16_atten_fusion/' + id[0][-15:])

                elif eval_scales != None and mode == 'add_fusion':
                    pred = model(img, scales=eval_scales, eval_mode='add_fusion').argmax(dim=1)
                else:
                    # scale_factor = 0.5
                    # h, w = img.shape[-2:]
                    # new_h = int(h * scale_factor + 0.5)
                    # new_w = int(w * scale_factor + 0.5)
                    # if scale_factor != 1.0:
                    #     img = F.interpolate(img, size=[new_h, new_w], mode='bilinear', align_corners=False)
                    pred = model(img)
                    # if scale_factor != 1.0:
                        # pred = F.interpolate(pred, size=[h, w], mode='bilinear', align_corners=False)
                    pred = pred.argmax(dim=1)
                    # pred = F.interpolate(pred, size=[h, w], mode='nearest')
                    # rgb_pre = to_rgb(cmap, pred.cpu())
                    # result = Image.fromarray(rgb_pre)
                    # result.save('/data/users/zhengzhiyu/SSL/SSSS_LL/UniMatch-main/prediction_result/1_16/scalematch_pascal_1_16_1.0/' + id[0][-15:])

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU, iou_class





if __name__ == '__main__':

    # evalutate_batch_iou()

    set_random_seed(1234)
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        # writer = SummaryWriter(args.save_path)

        # os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    checkpoint = torch.load((args.ckpt_path), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)
    # eval_mode = 'add_fusion'
    eval_mode = 'original'
    # eval_mode = 'sliding_window'
    # eval_scales=[0.5, 1.0, 2.0]
    mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, eval_scales=None)

    if rank == 0:
        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                        'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
        logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))



    # eval_mode = 'add'
    # mIoU_mscales, iou_class_mscales = evaluate(model, valloader, eval_mode, cfg, eval_scales=[0.5, 1.0, 2.0])
    # if rank == 0:
    #     for (cls_idx, iou) in enumerate(iou_class_mscales):
    #         logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
    #                     'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
    #     logger.info('***** Evaluation {} ***** >>>> MeanIoU_Mscales: {:.2f}\n'.format(eval_mode, mIoU_mscales))
    #
    # eval_mode = 'atten_fusion'
    # mIoU_mscales, iou_class_mscales = evaluate(model, valloader, eval_mode, cfg, eval_scales=[0.25, 0.5, 1.0, 2.0])
    #
    # if rank == 0:
    #     for (cls_idx, iou) in enumerate(iou_class_mscales):
    #         logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
    #                     'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
    #     logger.info('***** Evaluation {} ***** >>>> MeanIoU_Mscales: {:.2f}\n'.format(eval_mode, mIoU_mscales))

