import sys
import os
# set working directory to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch
import torch.utils.data
import utils
from bert.modeling_bert import BertModel
from lib import segmentation
from PIL import Image
import json
from pycocotools.coco import COCO
from tasks.code.metric.cidereval.eval import CIDErEvalCap
from data.DiverseDataset import DiverseDataset
from tasks.code.model import RemoteSAM
import warnings
warnings.filterwarnings("ignore")


def evaluate(gtjson, predjson):
    cocoGt = COCO(gtjson)
    cocoDt = cocoGt.loadRes(predjson)
    cocoeval_cap = CIDErEvalCap(cocoGt, cocoDt)
    cocoeval_cap.evaluate()

    print("\n########## Evaluation Summary ##########")
    print("CIDEr:\t{:.3f}%".format(cocoeval_cap.eval['CIDEr'] * 100.))
    print()


def infer(model, data_loader, bert_model, args):
    model = RemoteSAM(model, args.local_rank, use_EPOC=args.EPOC, EPOC_threshold = 0.25)
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_total = 0
    json_file = 'result.json'
    gt_json = {"images": [], "annotations": []}
    pred_json = []
    classnames = data_loader.dataset.classes

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, 'Test:'):
            origin_image, groundtruth, image_name = data

            groundtruth = [g[0] for g in groundtruth]
            image_name = image_name[0]

            result = model.captioning(image=origin_image.squeeze(0).numpy(), classnames=classnames)
            # result = result.split(".")[0] + '.'

            # to form json text
            pred_json.append({
                "image_id": data_loader.dataset.image2id[image_name],
                "caption": result
            })

            # to form groundtruth json
            gt_json['images'].append({
                "id": data_loader.dataset.image2id[image_name],
                "file_name": image_name,
                "height": origin_image.shape[1],
                "width": origin_image.shape[2]
            })
            for caption in groundtruth:
                gt_json['annotations'].append({ 
                    "id": len(gt_json['annotations']), 
                    "image_id": data_loader.dataset.image2id[image_name], 
                    "caption": caption
                })

            cum_total += 1

            del groundtruth, origin_image, image_name

            #! only for debug
            # if cum_total == 100:
            #     break

    # sync ddp processes
        # gt_json['images']
    gathered = [None for _ in range(utils.get_world_size())]
    torch.distributed.all_gather_object(gathered, gt_json['images'])
    gt_json['images'] = sum(gathered, []) 
        # gt_json['annotations']
    gathered = [None for _ in range(utils.get_world_size())]
    torch.distributed.all_gather_object(gathered, gt_json['annotations'])
    gt_json['annotations'] = sum(gathered, []) 
        # pred_json
    gathered = [None for _ in range(utils.get_world_size())]
    torch.distributed.all_gather_object(gathered, pred_json)
    pred_json = sum(gathered, []) 

    if args.local_rank == 0:
        gt_path = os.path.join(args.save_path, 'groundtruth.json')
        pred_path = os.path.join(args.save_path, 'result.json')
        # save caption
        pred_json.sort(key=lambda x: x['image_id'])
        with open(pred_path, 'w') as f:
            json.dump(pred_json, f, indent=4)
        # save groundtruth
        gt_json['images'].sort(key=lambda x: x['id'])
        gt_json['annotations'].sort(key=lambda x: x['image_id'])
        with open(gt_path, 'w') as f:
            json.dump(gt_json, f, indent=4)

        evaluate(gt_path, pred_path)


def main(args):
    utils.init_distributed_mode(args)

    dataset_test = DiverseDataset(args)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, 
        num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=False)
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
        batch_size=1, sampler=test_sampler, num_workers=args.workers, pin_memory=args.pin_mem)
    
    model = segmentation.__dict__[args.model](pretrained='',args=args)
    model.to(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    single_model = model.module 

    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'], strict=False)

    if args.model != 'lavt_one':
        model_class = BertModel
        bert_model = model_class.from_pretrained(args.ck_bert)
        if args.ddp_trained_weights:
            bert_model.pooler = None
        bert_model.to(args.local_rank)
        bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
        single_bert_model = bert_model
        single_bert_model.load_state_dict(checkpoint['bert_model'])
    else:
        bert_model = None
        single_bert_model = None

    infer(model, data_loader_test, bert_model, args=args)

    


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    main(args)