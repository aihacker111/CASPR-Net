"""
Train and eval functions used in main.py
"""
import math
import sys
from contextlib import nullcontext
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

def set_bn_state(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    set_bn_eval=False,
                    # DINOv3-style per-step scheduler
                    lr_schedule=None, wd_schedule=None, last_layer_lr_schedule=None,
                    step_offset: int = 0, amp: bool = True,
                    accumulation_steps: int = 1,):
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be at least 1")

    model.train(set_training_mode)
    if set_bn_eval:
        set_bn_state(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('wd', utils.SmoothedValue(
        window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    num_batches = len(data_loader)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (samples, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        accumulation_start = (batch_idx // accumulation_steps) * accumulation_steps
        accumulation_size = min(
            accumulation_steps, num_batches - accumulation_start)
        update_grad = (
            (batch_idx + 1) % accumulation_steps == 0
            or batch_idx + 1 == num_batches
        )

        # ── DINOv3 per-step LR/WD update ─────────────────────────────────
        if lr_schedule is not None and batch_idx == accumulation_start:
            it = step_offset + batch_idx // accumulation_steps
            lr  = lr_schedule[it]
            wd  = wd_schedule[it]
            ll_lr = last_layer_lr_schedule[it]
            for pg in optimizer.param_groups:
                pg["weight_decay"] = wd * pg["wd_multiplier"]
                if pg["is_last_layer"]:
                    pg["lr"] = ll_lr * pg["lr_multiplier"]
                else:
                    pg["lr"] = lr * pg["lr_multiplier"]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # DDP would otherwise all-reduce every micro-batch. Synchronize only
        # when the accumulated gradient is about to be applied.
        sync_context = (
            nullcontext()
            if update_grad or not hasattr(model, 'no_sync')
            else model.no_sync()
        )
        with sync_context:
            with torch.autocast(device_type="cuda", enabled=amp):
                outputs = model(samples)
                loss = criterion(samples, outputs, targets)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # Average over the current accumulation window. The final window
            # may contain fewer micro-batches than accumulation_steps.
            backward_loss = loss / accumulation_size

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(
                backward_loss, optimizer,
                clip_grad=clip_grad,
                clip_mode=clip_mode,
                parameters=model.parameters(),
                create_graph=is_second_order,
                need_update=update_grad,
            )

        if update_grad:
            optimizer.zero_grad(set_to_none=True)

        if model_ema is not None and update_grad:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
