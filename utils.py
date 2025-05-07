from __future__ import print_function
from collections import defaultdict, deque
import datetime
import math
import time
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch import nn

import errno
import os

import sys

import numpy as np
import cv2


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), f"Key '{k}' must be float/int, got {type(v)} with value {v}"
            self.meters[k].update(v)



    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        print(iterable)
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
            'max mem: {memory:.0f}'
        ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / MB))
                sys.stdout.flush()

            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environment: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    setup_for_distributed(is_main_process())

    if args.output_dir:
        mkdir(args.output_dir)
    if args.model_id:
        mkdir(os.path.join('./models/', args.model_id))


def M2B_wEPOC(ori_mask, new_mask, probs, area_threshold, conf_threshold, box_type):

    _, label_mtrix_new, stats_new, _ = cv2.connectedComponentsWithStats(new_mask, connectivity=8)
    num_connect, label_mtrix, _, _ = cv2.connectedComponentsWithStats(ori_mask, connectivity=8)

    reserve_indices = []
    for i in range(1, num_connect):
        covered_new = np.where(label_mtrix == i, label_mtrix_new, 0)
        covered_labels = np.unique(covered_new)
        covered_labels = covered_labels[covered_labels != 0]
        if len(covered_labels) == 0:
            continue

        # find max area
        areas = [stats_new[j][-1] for j in covered_labels]
        max_area = max(areas)

        for j in covered_labels:
            if stats_new[j][-1] > max_area * area_threshold:
                reserve_indices.append(j)

    boxes = []
    for i in reserve_indices:
        # confidence
        box_mask = np.where(label_mtrix_new == i, 1, 0).astype(np.uint8)
        conf = np.sum(probs * box_mask)/stats_new[i][-1]
        if conf < conf_threshold:
            continue
        # bnbox
        if box_type == 'hbb':
            x, y, w, h, _ = stats_new[i]
            boxes.append([x, y, x+w, y+h, conf])

        elif box_type == 'obb':
            countour, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rec = cv2.minAreaRect(countour[0])
            box = cv2.boxPoints(rec).astype(int)
            boxes.append([box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1], box[3][0], box[3][1], conf])

    return boxes


def M2B_woEPOC(mask, probs, conf_threshold, box_type):

    boxes = []
    num_connect, label_mtrix, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    for i in range(1, num_connect):
        # confidence
        box_mask = np.where(label_mtrix == i, 1, 0).astype(np.uint8)
        conf = np.sum(probs * box_mask)/stats[i][-1]
        if conf < conf_threshold:
            continue
        # bnbox
        if box_type == 'hbb':
            x, y, w, h, _ = stats[i]
            boxes.append([x, y, x+w, y+h, conf])
        elif box_type == 'obb':
            countour, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rec = cv2.minAreaRect(countour[0])
            box = cv2.boxPoints(rec).astype(int)
            boxes.append([box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1], box[3][0], box[3][1], conf])
            
    return boxes


def box1_in_box2(box1, box2):

    # cv2 implementation, change to shapely recommended
    assert len(box1) == 5 or len(box1) == 9, f"len(box) should be 5 or 9, but got {len(box1)}"
    assert len(box2) == 5 or len(box2) == 9, f"len(box) should be 5 or 9, but got {len(box2)}"

    if len(box1) == 5:
        # expand to 4 points
        x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3]
        box1 = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        x1, y1, x2, y2 = box2[0], box2[1], box2[2], box2[3]
        box2 = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    else:
        box1 = np.array(box1[:-1]).reshape(-1, 2)
        box2 = np.array(box2[:-1]).reshape(-1, 2)

    w = max(box1[:, 0].max(), box2[:, 0].max())
    h = max(box1[:, 1].max(), box2[:, 1].max())

    canvas1 = np.zeros((h, w), dtype=np.uint8)
    canvas2 = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(canvas1, [box1], 1)
    cv2.fillPoly(canvas2, [box2], 1)

    return np.logical_and(canvas1, canvas2).sum() == canvas1.sum()


def M2B(mask, probs, *,
        box_type,
        new_mask=None,
        area_threshold=0.2,
        conf_threshold=0.01,
        remove_inner=True
        ):
    """
    Convert mask to bounding box.
    Args:
        :param mask: original mask from model
        :param probs: probability map from model
        :param box_type: 'hbb' or 'obb'
        :param new_mask: mask after EPOC, default is None
        :param area_threshold: only when new_mask is specified, this parameter is used to filter out small masks
        :param conf_threshold: confidence threshold
        :param remove_inner: whether to remove inner boxes
    Returns:
        :return: boxes: list of bounding boxes
    """

    assert box_type in ['hbb', 'obb'], f"box_type should be 'hbb' or 'obb', but got {box_type}"

    if new_mask is not None:
        boxes = M2B_wEPOC(mask, new_mask, probs, area_threshold, conf_threshold, box_type)
    else:
        boxes = M2B_woEPOC(mask, probs, conf_threshold, box_type)

    # remove small box contained in large box
    if remove_inner:
        indice = []
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i == j:
                    continue
                if box1_in_box2(boxes[i], boxes[j]):
                    indice.append(i)
        boxes = [boxes[i] for i in range(len(boxes)) if i not in indice]

    return boxes


# For EPOC Inference
# From: https://arxiv.org/abs/2402.14327

def generate_crop_boxes(im_size, n_layers=1, overlap=0):
    """
    Generates a list of crop boxes of different sizes. Each layer
    has (2**i)**2 boxes for the ith layer.
    """

    crop_boxes, layer_idxs = [], []
    im_w , im_h = im_size

    # Original image
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        # overlap = int(overlap_ratio * min(im_h, im_w) * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # Crops in XYWH format
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    # only keep layer_id=n_layers
    crop_boxes = [box for box, layer in zip(crop_boxes, layer_idxs) if layer == n_layers]
    layer_idxs = [layer for layer in layer_idxs if layer == n_layers]

    return crop_boxes


def boundary_zero_padding(probs, p=15):
    # from https://arxiv.org/abs/2308.13779
    
    zero_p = p//3
    alpha_p = zero_p*2
    
    probs[:, :alpha_p] *= 0.5
    probs[:, -alpha_p:] *= 0.5
    probs[:alpha_p, :] *= 0.5
    probs[-alpha_p:, :] *= 0.5

    probs[:, :zero_p] = 0
    probs[:, -zero_p:] = 0
    probs[:zero_p, :] = 0
    probs[-zero_p:, :] = 0

    return probs


def EPOC(image, image_processor, model, pyramid_layers=0, overlap=90, resolution=None):

    if resolution:
        image_processor.size['height'] = resolution
        image_processor.size['width'] = resolution

    def run(image, bzp=0):
        encoding = image_processor(image, return_tensors="pt")
        pixel_values = encoding.pixel_values.to(model.device).to(model.dtype)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        logits = outputs.logits.float().cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        probs = torch.sigmoid(upsampled_logits)[0, 0].detach().numpy()

        if bzp>0:
            probs = boundary_zero_padding(probs, p=bzp)
        return probs
    
    global_probs = run(image)

    if pyramid_layers > 0:
        for layer in range(1, pyramid_layers+1):
            boxes = generate_crop_boxes(image.size, n_layers=layer, overlap=overlap)
            for box in boxes:
                x1, y1, x2, y2 = box
                crop = image.crop(box)
                probs = run(crop, bzp=overlap)
                global_probs[y1:y2, x1:x2] += probs
        global_probs /= (pyramid_layers + 1)

    return global_probs
