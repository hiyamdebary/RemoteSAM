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
from data.DiverseDataset import DiverseDataset
from tasks.code.model import RemoteSAM
import warnings
warnings.filterwarnings("ignore")


def infer(model, data_loader, bert_model, args):
    model = RemoteSAM(model, args.local_rank, use_EPOC=args.EPOC, EPOC_threshold = 0.25)
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_total = 0
    cum_boxes = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, 'Test:'):
            origin_image, groundtruth, image_name = data

            groundtruth = groundtruth.squeeze(0).numpy()
            image_name = image_name[0]

            boxes = model.detection(image=origin_image.squeeze(0).numpy(), classnames=[args._class])
            if boxes[args._class]:
                boxes = boxes[args._class]

                # to form txt
                if len(boxes) > 0:
                    for i in range(len(boxes)):
                        temp = boxes[i]
                        temp.append(image_name)
                        cum_boxes.append(temp)

            cum_total += 1

            del groundtruth, origin_image, image_name, boxes

            #! only for debug
            # if cum_total == 10:
            #     break

    # sync ddp processes
        # cum_boxes
    gathered = [None for _ in range(utils.get_world_size())]
    torch.distributed.all_gather_object(gathered, cum_boxes)
    cum_boxes = sum(gathered, []) 

    cum_boxes.sort(key=lambda x: x[5])

    if args.local_rank == 0:
        # save result
        with open(args.save_path + 'Task2_' + args._class + '.txt', 'w') as f:
            for box in cum_boxes:
                x1, y1, x2, y2, conf, image_name = box
                pts = [x1, y1, x2, y1, x2, y2, x1, y2]
                box_string = ' '.join([str(float(a)) for a in pts])
                f.write(image_name + " " + str(conf) + " " + box_string + '\n')

    print("Done evaluating the category: ", args._class)
    print()


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