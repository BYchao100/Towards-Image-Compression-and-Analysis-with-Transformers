"""
Train and eval functions used in main.py
"""
import math
import sys
import os
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import JointLoss, DenormalizedMSELoss
import utils


def train_one_epoch(model: torch.nn.Module, criterion: JointLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, aux_optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        outputs, aux_loss = model(samples)
        loss = criterion(samples, outputs, targets)
        loss_value = loss.item()
        loss.backward()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if max_norm is not None and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        aux_loss_value = aux_loss.item()
        aux_loss.backward()
        aux_optimizer.step()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(aux_loss=aux_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, output_dir, write_img=True):
    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_rec = DenormalizedMSELoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        B, _, H, W = images.shape
        num_pixels = B * H * W

        # compute output
        output, _ = model(images)
        loss_cls = criterion_cls(output[0], target)
        loss_rec = criterion_rec(output[1], images)
        loss_bpp = (torch.log(output[2]).sum() + torch.log(output[3]).sum()) / (-math.log(2) * num_pixels)
        psnr = utils.img_distortion(output[1], images)

        if write_img:
            utils.imwrite(images[:4], os.path.join(output_dir,'example_org.png'))
            utils.imwrite(output[1][:4], os.path.join(output_dir,'example_rec.png'))
            write_img = False

        acc1, acc5 = accuracy(output[0], target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss_cls=loss_cls.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['loss_rec'].update(loss_rec.item(), n=batch_size)
        metric_logger.meters['loss_bpp'].update(loss_bpp.item(), n=batch_size)
        metric_logger.meters['psnr'].update(psnr, n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss_cls {losses_cls.global_avg:.3f} loss_rec {losses_rec.global_avg:.3f} loss bpp {losses_bpp.global_avg:.3f} psnr {psnr.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses_cls=metric_logger.loss_cls, losses_rec=metric_logger.loss_rec, losses_bpp=metric_logger.loss_bpp, psnr=metric_logger.psnr))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_real(data_loader, model, device, output_dir, write_img=True):
    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_rec = DenormalizedMSELoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    model.update(force=True)
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        B, _, H, W = images.shape
        num_pixels = B * H * W

        # compute output
        out_enc = model.compress(images)
        output = model.decompress(out_enc['strings'], out_enc['shape'])
        loss_cls = criterion_cls(output[0], target)
        loss_rec = criterion_rec(output[1], images)
        psnr = utils.img_distortion(output[1], images)

        bitstream = sum([len(out_enc['strings'][0][i]) + len(out_enc['strings'][1][i]) + 16 for i in range(B)])
        loss_bpp = bitstream * 8 / num_pixels

        if write_img:
            utils.imwrite(images[:4], os.path.join(output_dir,'example_org.png'))
            utils.imwrite(output[1][:4], os.path.join(output_dir,'example_rec.png'))
            write_img = False

        acc1, acc5 = accuracy(output[0], target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss_cls=loss_cls.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['loss_rec'].update(loss_rec.item(), n=batch_size)
        metric_logger.meters['loss_bpp'].update(loss_bpp, n=batch_size)
        metric_logger.meters['psnr'].update(psnr, n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss_cls {losses_cls.global_avg:.3f} loss_rec {losses_rec.global_avg:.3f} loss bpp {losses_bpp.global_avg:.3f} psnr {psnr.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses_cls=metric_logger.loss_cls, losses_rec=metric_logger.loss_rec, losses_bpp=metric_logger.loss_bpp, psnr=metric_logger.psnr))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
