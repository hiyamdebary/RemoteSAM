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


def eval(class_count, true, prob):
    from tasks.code.metric.RunningScore import MultiClassRunningScore
        
    prob = torch.Tensor(prob)
    true = torch.Tensor(true)

    running_metrics = MultiClassRunningScore(class_count)
    running_metrics.update(true.type(torch.int64), prob.type(torch.float64))
    
    accuracy = running_metrics.accuracy()

    # sumarize evaluation
    print("\n########## Evaluation Summary ##########")
    print("Accuracy: {:.3f}%".format(accuracy["Accuracy"] * 100.))
    print()


def classification(model, data_loader, bert_model, args):
    model = RemoteSAM(model, args.local_rank, use_EPOC=args.EPOC, EPOC_threshold = 0.25)
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_total = 0
    true = []
    prob = []
    classnames = data_loader.dataset.classes

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, 'Test:'):
            origin_image, groundtruth, image_name = data

            groundtruth = groundtruth.item()
            image_name = image_name[0]

            _, prob_ins = model.multi_class_cls(image=origin_image.squeeze(0).numpy(), classnames=classnames, return_prob=True)

            true.append(groundtruth)
            prob.append([prob_ins[classname] for classname in classnames])

            cum_total += 1

            del groundtruth, origin_image, image_name, prob_ins

            #! only for debug
            # if cum_total == 10:
            #     break

    # sync the process
        # true
    gathered = [None for _ in range(utils.get_world_size())]
    torch.distributed.all_gather_object(gathered, true)
    true = sum(gathered, []) 
        # prob
    gathered = [None for _ in range(utils.get_world_size())]
    torch.distributed.all_gather_object(gathered, prob)
    prob = sum(gathered, []) 

    return len(classnames), true, prob


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

    class_count, true, prob = classification(model, data_loader_test, bert_model, args=args)

    eval(class_count, true, prob)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    main(args)