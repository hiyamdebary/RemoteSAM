import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random
import transformers

from bert.tokenization_bert import BertTokenizer
import pickle
import h5py
import json
from refer.refer import REFER
from tqdm import tqdm
from args import get_parser
import cv2
from torch.utils.data import Dataset
from pycocotools import mask

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()


def add_random_boxes(img, min_num=20, max_num=60, size=32):
    h,w = size, size
    img = np.asarray(img).copy()
    img_size = img.shape[1]
    boxes = []
    num = random.randint(min_num, max_num)
    for k in range(num):
        y, x = random.randint(0, img_size-w), random.randint(0, img_size-h)
        img[y:y+h, x: x+w] = 0
        boxes. append((x,y,h,w) )
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    return img


class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)

        self.max_tokens = 20

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        num_images_to_mask = int(len(ref_ids) * 0.2)
        self.images_to_mask = random.sample(ref_ids, num_images_to_mask)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        self.eval_mode = eval_mode
        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            attentions_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name']))
        if self.split == 'train' and this_ref_id in self.images_to_mask:
            img = add_random_boxes(img)

        ref = self.refer.loadRefs(this_ref_id)

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            img, target = self.image_transforms(img, annot)

        if self.eval_mode:
            embedding = []
            att = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                a = self.attention_masks[index][s]
                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))

            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
        else:
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]

        return img, target, tensor_embeddings, attention_mask


class All_Dataset(data.Dataset):
    def __init__(self,
                 args,
                 image_transforms=None,
                 max_tokens=20,
                 split='train',
                 eval_mode=True,
                 logger=None) -> None:
        """
        parameters:
            args: argparse obj
            image_transforms: transforms apply to image and mask
            max_tokens: determined the max length of token
            split: ['train','val','testA','testB']
            eval_mode: whether in training or evaluating
        """

        self.classes = []
        self.image_transforms = image_transforms
        self.split = split
        self.args = args
        self.eval_mode = eval_mode
        self.max_tokens = max_tokens
        self.image_path = args.imageFolder
        self.ref_file = args.ref_file_path
        self.instances_path=args.instances_path
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

        self.maskToSentence,self.maskToimageName,self.maskTosplit,self.maskToCategory,self.maskToDecodedRLE,self.masktoArea,self.list_mask = self.load_captions()
        self.refToInput,self.refToAttention,self.refToExist=self.get_sent_embeddings()
        if logger:
            logger.info(f"=> loaded successfully '{args.dataset}', split by {args.splitBy}, split {split}")
        #load ref ,find caption ,image,according to mask
    def load_captions(self):

        maskToSentence = {}
        maskToimageName = {}
        maskTosplit = {}
        maskToCategory = {}
        masktoRLE={}
        masktoArea={}
        maskToDecodedRLE={}
        list_mask=[]
        with open(self.ref_file, 'rb') as f:
            data = pickle.load(f)
            total_lines = len(data)  
            pbar = tqdm(data, desc="Loading captions", total=total_lines)  
            for line in pbar:
                split_line = line["split"]
                ann_id = line["ann_id"]
                img_name = line["file_name"]
                img_id = line["image_id"]
                category_id = line["category_id"]
                if split_line == self.split :
                    list_mask.append(ann_id)


                    maskToSentence[ann_id] = line["sentences"][0]["sent"]
                    maskToimageName[ann_id]=img_name
                    maskTosplit[ann_id]=split_line
                    maskToCategory[ann_id]=category_id

                # Only load the ann_id of the corresponding training set/validation set/test set.
            pbar.close()  
        with open(self.instances_path, 'r') as fi:
            data_json = json.load(fi)
            annotations = data_json['annotations']
            total_annotations = len(annotations)  
            pbar = tqdm(annotations, desc="Loading instance data", total=total_annotations)  
            for data in pbar:
                ann_id = data["id"]

                rle = data['segmentation']
                # decoded_mask = mask.decode(rle)  
                maskToDecodedRLE[ann_id] = rle

                masktoArea[ann_id] = data['area']
            pbar.close() 

        return maskToSentence,maskToimageName,maskTosplit,maskToCategory,maskToDecodedRLE,masktoArea,list_mask
    def get_sent_embeddings(self):
        refToInput={}
        refToAttention={}
        refToExist={}
        total_mask_ids = len(self.list_mask)
        pbar = tqdm(self.list_mask, desc="Generating sentence embeddings", total=total_mask_ids)
        for mask_id in pbar:
            attention_mask = [0] * self.max_tokens
            padded_input_id = [0] * self.max_tokens
            ref=self.maskToSentence[mask_id]
            input_id=self.tokenizer.encode(text=ref, add_special_tokens=True)
            input_id = input_id[:self.max_tokens]

            padded_input_id[:len(input_id)] = input_id
            attention_mask[:len(input_id)] = [1] * len(input_id)

            input_id=torch.tensor(padded_input_id).unsqueeze(0)
            attention_mask=torch.tensor(attention_mask).unsqueeze(0)
            exist = torch.Tensor([True])
            refToInput[mask_id]=input_id
            refToAttention[mask_id]=attention_mask
            refToExist[mask_id]=exist
        pbar.close()
        return refToInput,refToAttention,refToExist
    def get_exist(self, ref, sent_index):
        if "exist" in ref["sentences"][sent_index].keys():
            exist = torch.Tensor([ref["sentences"][sent_index]["exist"]])
        else:
            exist = torch.Tensor([True])
        return exist

    def __len__(self):
        return len(self.list_mask)

    def __getitem__(self, index):
        mask_id=self.list_mask[index]
        img_name = self.maskToimageName[mask_id]
        # ref=self.maskToSentence[mask_id]
        # split=self.maskTosplit[mask_id]
        img = Image.open(os.path.join(self.image_path, img_name)).convert("RGB")
        rle = self.maskToDecodedRLE[mask_id]
        m = mask.decode(rle) 

        m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
        ref_mask = m.astype(np.uint8)  # convert to np.uint8
        # compute area


        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1
        # convert it to a Pillow image
        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        # tensor_embeddings,attention_mask,exist = self.get_sent_embeddings(ref)
        tensor_embeddings=self.refToInput[mask_id]
        tensor_embeddings=tensor_embeddings.permute(1, 0)
        attention_mask=self.refToAttention[mask_id]
        attention_mask = attention_mask.permute(1, 0)
        # attention_mask = attention_mask.unsqueeze(0)
        
        # print("attention_mask:", attention_mask.shape)
        exist=self.refToExist[mask_id]

        if self.image_transforms is not None:
            # involves transform from PIL to tensor and mean and std normalization
            img, target = self.image_transforms(img, annot)
        else:
            target = annot
        # if self.eval_mode:
        #     print("eval")
        #     tensor_embeddings=tensor_embeddings.unsqueeze(2)
            # attention_mask=attention_mask.unsqueeze(2)
            # exist=exist.unsqueeze(1)


        return img, target, tensor_embeddings, attention_mask, exist