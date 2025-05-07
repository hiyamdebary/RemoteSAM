import sys
import os
# set working directory to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch
import torch.utils.data
import utils
import numpy as np
from bert.modeling_bert import BertModel
from lib import segmentation
from data.DiverseDataset import DiverseDataset
from tasks.code.model import RemoteSAM
import warnings
warnings.filterwarnings("ignore")


def infer(model, data_loader, bert_model, device):
    model = RemoteSAM(model, device, use_EPOC=args.EPOC, EPOC_threshold = 0.25)
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_correct, cum_total = 0, 0
    cum_IoU = []
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros_like(eval_seg_iou_list, dtype=np.int32)

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, 'Test:'):
            origin_image, groundtruth, image_name, sentence = data

            # prepare
            groundtruth = groundtruth.squeeze(0).numpy()
            image_name = image_name[0]
            sentence = sentence[0]

            boxes = model.visual_grounding(image=origin_image.squeeze(0).numpy(), sentence=sentence)

            if boxes:
                IoU = computeIoU(boxes, groundtruth)
                cum_IoU.append(IoU)
            else:
                cum_IoU.append(0.0)

            # evaluate
            for n_eval_iou in range(len(eval_seg_iou_list)):
                if IoU >= eval_seg_iou_list[n_eval_iou]:
                    seg_correct[n_eval_iou] += 1

            cum_total += 1

            del groundtruth, origin_image, image_name, boxes
            if bert_model is not None:
                del last_hidden_states, embedding

            #! only for debug
            # if cum_total == 10:
            #     break

    # sumarize evaluation
    print("\n########## Evaluation Summary ##########")
    print('Mean IoU:\t{:.2f}%'.format(np.mean(np.array(cum_IoU)) * 100.))
    for n_eval_iou in range(len(eval_seg_iou_list)):
        print('precision@{}:\t{:.2f}%'.format(eval_seg_iou_list[n_eval_iou], seg_correct[n_eval_iou] / cum_total * 100.))
    print()


def computeIoU(pred, gt):
    pred_x1, pred_y1, pred_x2, pred_y2 = pred
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    # calculate intersection
    inter_x1 = max(pred_x1, gt_x1)
    inter_y1 = max(pred_y1, gt_y1)
    inter_x2 = min(pred_x2, gt_x2)
    inter_y2 = min(pred_y2, gt_y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # calculate union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union_area = pred_area + gt_area - inter_area

    if union_area == 0:
        IoU = 0.0
    else:
        IoU = inter_area / union_area

    return IoU


def main(args):
    device = torch.device(args.device)

    dataset_test = DiverseDataset(args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, 
        batch_size=1, sampler=test_sampler, num_workers=args.workers)
    
    single_model = segmentation.__dict__[args.model](pretrained='',args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'], strict=False)
    model = single_model.to(device)

    if args.model != 'lavt_one':
        model_class = BertModel
        single_bert_model = model_class.from_pretrained(args.ck_bert)
        if args.ddp_trained_weights:
            single_bert_model.pooler = None
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)
    else:
        bert_model = None

    infer(model, data_loader_test, bert_model, device=device)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    main(args)