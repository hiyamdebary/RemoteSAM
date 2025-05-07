import sys
import os
# set working directory to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torchvision
import numpy as np
import utils
import transformers
from lib import segmentation
import transforms as T
from PIL import Image
from .RuleBasedCaptioning import single_captioning


def get_transform():
    transforms = [
                  T.Resize(896, 896),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]
    return T.Compose(transforms)


def embed_sentences(sentences, max_tokens=20):
    # init
    bert_model = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = []
    attentions = []

    for sentence in sentences:
        sentence_tokenized = bert_model.encode(text=sentence, add_special_tokens=True)
        sentence_tokenized = sentence_tokenized[:max_tokens]  
        # pad the tokenized sentence
        padded_sent_toks = [0] * max_tokens
        padded_sent_toks[:len(sentence_tokenized)] = sentence_tokenized
        # create a sentence token mask: 1 for real words; 0 for padded tokens
        attention_mask = [0] * max_tokens
        attention_mask[:len(sentence_tokenized)] = [1]*len(sentence_tokenized)
        inputs.append([padded_sent_toks])
        attentions.append([attention_mask])
    
    return torch.tensor(inputs), torch.tensor(attentions)


def init_demo_model(checkpoint, device):
    from args import get_parser
    args = get_parser().parse_args()
    args.device = device
    args.window12 = True

    model = segmentation.__dict__["lavt_one"](pretrained='', args=args)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'], strict=False)
    model = model.to(device)
    return model


class RemoteSAM():

    RemoteSAM_model = None
    EPOC_model = None
    image_processor = None
    EPOC_threshold = None

    MLC_balance_factor = 0.5
    MCC_balance_factor = 1.0
    GMP = torch.nn.AdaptiveMaxPool2d(1)
    GAP = torch.nn.AdaptiveAvgPool2d(1)

    def __init__(self, RemoteSAM_model, device, *, use_EPOC, EPOC_threshold=0.25, MLC_balance_factor=0.5, MCC_balance_factor=1.0):
        self.RemoteSAM_model = RemoteSAM_model
        self.RemoteSAM_model.eval()
        self.device = device
        self.EPOC_threshold = EPOC_threshold
        self.MLC_balance_factor = MLC_balance_factor
        self.MCC_balance_factor = MCC_balance_factor
        if use_EPOC:
            EPOC_checkpoint = "chendelong/DirectSAM-EntitySeg-1024px-0501"
            self.image_processor = transformers.AutoImageProcessor.from_pretrained(EPOC_checkpoint, reduce_labels=True)
            EPOC_model = transformers.AutoModelForSemanticSegmentation.from_pretrained(EPOC_checkpoint, num_labels=1, ignore_mismatched_sizes=True)
            self.EPOC_model = EPOC_model.to(device).eval()


    def check_input(self, image, classnames):

        assert isinstance(image, Image.Image) or isinstance(image, np.ndarray), "image should be PIL Image or ndarray"
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if isinstance(classnames, str):
            classnames = [classnames]
        else:
            assert isinstance(classnames, list), "classnames should be str or list"

        return image, classnames

    
    def semantic_seg(self, *, image, classnames, return_prob=False):
        image, classnames = self.check_input(image, classnames)

        mask_result = {}
        prob_result = {}

        origin_image = image
        image, _ = get_transform()(image, image)
        image = image.unsqueeze(0).to(self.device)

        inputs, attentions = embed_sentences([f"{classname} in the image" for classname in classnames])
        inputs = inputs.to(self.device)
        attentions = attentions.to(self.device)

        for j in range(inputs.shape[0]):

            output = self.RemoteSAM_model(image, inputs[j], l_mask=attentions[j])

            mask = output.cpu().argmax(1, keepdim=True)  # (1, 1, resized_shape)
            mask = torch.nn.functional.interpolate(mask.float(), origin_image.size[::-1])  # (1, 1, origin_shape)
            mask = mask.squeeze().data.numpy().astype(np.uint8) # np(origin_shape)
            
            prob = torch.softmax(output, dim=1)[:, [1], :, :].cpu() # (1, 1, resized_shape)
            prob = torch.nn.functional.interpolate(prob.float(), origin_image.size[::-1])  # (1, 1, origin_shape)
            prob = prob.squeeze().data.numpy() # np(origin_shape)

            mask_result[classnames[j]] = mask
            prob_result[classnames[j]] = prob
            
        if return_prob:
            return mask_result, prob_result
        else:
            return mask_result


    def referring_seg(self, *, image, sentence, return_prob=False):
        image, sentence = self.check_input(image, sentence)
        assert len(sentence)==1

        origin_image = image
        image, _ = get_transform()(image, image)
        image = image.unsqueeze(0).to(self.device)

        inputs, attentions = embed_sentences(sentence)
        inputs = inputs[0].to(self.device)
        attentions = attentions[0].to(self.device)

        output = self.RemoteSAM_model(image, inputs, l_mask=attentions)

        mask = output.cpu().argmax(1, keepdim=True)  # (1, 1, resized_shape)
        mask = torch.nn.functional.interpolate(mask.float(), origin_image.size[::-1])  # (1, 1, origin_shape)
        mask = mask.squeeze().data.numpy().astype(np.uint8) # np(origin_shape)

        prob = torch.softmax(output, dim=1)[:, [1], :, :].cpu() # (1, 1, resized_shape)
        prob = torch.nn.functional.interpolate(prob.float(), origin_image.size[::-1])  # (1, 1, origin_shape)
        prob = prob.squeeze().data.numpy() # np(origin_shape)

        if return_prob:
            return mask, prob
        else:
            return mask


    def detection(self, *, image, classnames):
        image, classnames = self.check_input(image, classnames)

        masks, probs = self.semantic_seg(image=image, classnames=classnames, return_prob=True)

        result = {}

        origin_image = image
        image, _ = get_transform()(image, image)
        image = image.unsqueeze(0).to(self.device)

        if self.EPOC_model:
            countour = utils.EPOC(origin_image, self.image_processor, self.EPOC_model)
            binarilized = (countour > self.EPOC_threshold).astype(np.uint8)

        for classname in classnames:
            result[classname] = None

            if self.EPOC_model:
                refined_mask = masks[classname] * (1 - binarilized)
                boxes = utils.M2B(masks[classname], probs[classname], new_mask=refined_mask, box_type='hbb')
            else:
                boxes = utils.M2B(masks[classname], probs[classname], box_type='hbb')

            if len(boxes) > 0:
                indice = torchvision.ops.nms(torch.tensor([[a[0], a[1], a[2], a[3]] for a in boxes]).float(), torch.tensor([a[4] for a in boxes]).float(), 0.5)
                boxes = [boxes[i] for i in indice]
                result[classname] = boxes
        
        return result


    def counting(self, *, image, classnames):
        boxes = self.detection(image=image, classnames=classnames)
        result = {}
        for classname in classnames:
            if boxes[classname]:
                result[classname] = len(boxes[classname])
            else:
                result[classname] = 0
        return result


    # for a faster inference
    def __simple_forward(self, *, image, classnames):
        image, classnames = self.check_input(image, classnames)

        max_result = {}
        avg_result = {}

        origin_image = image
        image, _ = get_transform()(image, image)
        image = image.unsqueeze(0).to(self.device)

        inputs, attentions = embed_sentences([f"{classname} in the image" for classname in classnames])
        inputs = inputs.to(self.device)
        attentions = attentions.to(self.device)

        for j in range(inputs.shape[0]):

            output = self.RemoteSAM_model(image, inputs[j], l_mask=attentions[j])
            prob = torch.softmax(output, dim=1)[:, [1], :, :]

            max_result[classnames[j]] = self.GMP(prob).cpu().squeeze().data.numpy()
            avg_result[classnames[j]] = self.GAP(prob).cpu().squeeze().data.numpy()
            
        return max_result, avg_result


    def multi_label_cls(self, *, image, classnames, return_prob=False):
        image, classnames = self.check_input(image, classnames)

        max_result, avg_result = self.__simple_forward(image=image, classnames=classnames)

        scores = {}
        result = []
        for classname in classnames:
            score = self.MLC_balance_factor * max_result[classname] + (1 - self.MLC_balance_factor) * avg_result[classname]
            scores[classname] = score

            if score > 0.5:
                result.append(classname)
        
        if return_prob:
            return result, scores
        else:
            return result


    def multi_class_cls(self, *, image, classnames, return_prob=False):
        image, classnames = self.check_input(image, classnames)

        max_result, avg_result = self.__simple_forward(image=image, classnames=classnames)

        scores = {}
        for classname in classnames:
            score = self.MCC_balance_factor * max_result[classname] + (1 - self.MCC_balance_factor) * avg_result[classname]
            scores[classname] = score

        result = max(scores, key=lambda k: scores[k])
        if scores[result] < 0.5:
            result = None
        
        if return_prob:
            return result, scores
        else:
            return result


    def visual_grounding(self, *, image, sentence):
        image, sentence = self.check_input(image, sentence)

        mask, prob = self.referring_seg(image=image, sentence=sentence, return_prob=True)

        boxes = utils.M2B(mask, prob, box_type='hbb')

        if len(boxes) > 0:
            xmin = min(pred[0] for pred in boxes)
            ymin = min(pred[1] for pred in boxes)
            xmax = max(pred[2] for pred in boxes)
            ymax = max(pred[3] for pred in boxes)
            return [xmin, ymin, xmax, ymax]
        else:
            return None


    def captioning(self, *, image, classnames, region_split=4):
        temp = self.detection(image=image, classnames=classnames)

        # filter out empty boxes
        boxes = {}
        for classname in classnames:
            if temp[classname]:
                boxes[classname] = temp[classname]

        if isinstance(image, np.ndarray):
            shape = image.shape[:2]
        elif isinstance(image, Image.Image):
            shape = image.size[::-1]

        caption = single_captioning(boxes, shape, region_split)

        return caption


    

