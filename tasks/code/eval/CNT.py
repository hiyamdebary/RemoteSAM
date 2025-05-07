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
    model = RemoteSAM(model, args.local_rank, use_EPOC=args.EPOC, EPOC_threshold = 0.15)
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_correct, cum_total = 0, 0
    cum_boxes = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, 'Test:'):
            origin_image, groundtruth, image_name, classname, foo = data

            groundtruth = groundtruth.numpy()[0]
            image_name = image_name[0]
            foo = foo.numpy()[0]
            classname = classname[0]

            boxes = model.detection(image=origin_image.squeeze(0).numpy(), classnames=[classname])

            if boxes[classname]:
                boxes = boxes[classname]
                # prior_area_info filter
                indice = []
                for i in range(len(boxes)):
                    x1, y1, x2, y2, conf = boxes[i]
                    area = (x2-x1)*(y2-y1)
                    if area <= foo * 0.8:
                        continue
                    indice.append(i)
                boxes = [boxes[i] for i in indice]
                cnt = len(boxes)
            else:
                cnt = 0 
                
            # evaluate
            if cnt == groundtruth:
                cum_correct += 1

            cum_total += 1

            del groundtruth, origin_image, image_name, foo, boxes

            #! only for debug
            # if cum_total == 10:
            #     break

    # sync ddp processes
        # cum_total
    gathered = [None for _ in range(utils.get_world_size())]
    torch.distributed.all_gather_object(gathered, cum_total)
    cum_total = sum(gathered)
        # cum_correct
    gathered = [None for _ in range(utils.get_world_size())]
    torch.distributed.all_gather_object(gathered, cum_correct)
    cum_correct = sum(gathered)
    
    # sumarize evaluation
    print("\n########## Evaluation Summary ##########")
    print("Total count: {}".format(cum_total))
    print("Correct count: {}".format(cum_correct))
    print("Accuracy: {:.2f}%".format(cum_correct/cum_total * 100.))
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