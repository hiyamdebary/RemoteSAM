import os
import json

import cv2
from PIL import Image
import torch.utils.data as data
import transformers
from tqdm import tqdm  # 导入tqdm库用于显示进度条

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from pycocotools import mask

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
            total_lines = len(data)  # 获取总数据行数，用于进度条
            pbar = tqdm(data, desc="Loading captions", total=total_lines)  # 创建进度条对象
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
            pbar.close()  # 关闭进度条
        with open(self.instances_path, 'r') as fi:
            data_json = json.load(fi)
            annotations = data_json['annotations']
            total_annotations = len(annotations)  # 获取总注释数量，用于进度条
            pbar = tqdm(annotations, desc="Loading instance data", total=total_annotations)  # 创建进度条对象
            for data in pbar:
                ann_id = data["id"]

                rle = data['segmentation']
                # decoded_mask = mask.decode(rle)  # 假设mask.decode是解码RLE的函数
                maskToDecodedRLE[ann_id] = rle

                masktoArea[ann_id] = data['area']
            pbar.close()  # 关闭进度条

        return maskToSentence,maskToimageName,maskTosplit,maskToCategory,maskToDecodedRLE,masktoArea,list_mask
    def get_sent_embeddings(self):
        refToInput={}
        refToAttention={}
        refToExist={}
        total_mask_ids = len(self.list_mask)  # 获取mask_id的总数，用于进度条
        pbar = tqdm(self.list_mask, desc="Generating sentence embeddings", total=total_mask_ids)  # 创建进度条对象
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
        pbar.close()  # 关闭进度条
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
        try:
            img = Image.open(os.path.join(self.image_path, img_name)).convert("RGB")
        except (OSError, ValueError) as e:  
            # print(f"Error loading image {img_name}: {e}")  
            # 尝试更换扩展名  
            if img_name.lower().endswith('.png'):  
                new_img_name = img_name[:-4] + '.jpg'  # 替换为 .jpg 
                img = Image.open(os.path.join(self.image_path, new_img_name)).convert("RGB") 
            elif img_name.lower().endswith('.jpg'):  
                new_img_name = img_name[:-4] + '.png'  # 替换为 .png 
                img = Image.open(os.path.join(self.image_path, new_img_name)).convert("RGB") 
            else:  
                print("Unsupported file type.")  
                return None  

        rle = self.maskToDecodedRLE[mask_id]
        m = mask.decode(rle)  # 假设mask.decode是解码RLE的函数

        m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
        ref_mask = m.astype(np.uint8)  # convert to np.uint8
        # compute area


        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1
        # convert it to a Pillow image
        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        # tensor_embeddings,attention_mask,exist = self.get_sent_embeddings(ref)
        tensor_embeddings=self.refToInput[mask_id]
        attention_mask=self.refToAttention[mask_id]
        exist=self.refToExist[mask_id]

        if self.image_transforms is not None:
            # involves transform from PIL to tensor and mean and std normalization
            img, target = self.image_transforms(img, annot)
        else:
            target = annot
        if self.eval_mode:

            # tensor_embeddings=tensor_embeddings.unsqueeze(-1)
            # attention_mask=attention_mask.unsqueeze(-1)

            exist=exist.unsqueeze(1)


        return img, target, tensor_embeddings, attention_mask, exist

# from dataset.transform import get_transform
# import os
# import sys
# import torch.utils.data as data
# import torch
# import numpy as np
# from PIL import Image
# import transformers
# import argparse
#
# from torchvision import transforms
#
# def main():
#     # 模拟命令行参数，你可以根据实际需求调整这些参数值
#     args = argparse.Namespace(
#         imageFolder='/home/amax/yaoliang/rmsin/refer/data/images/rrsisd/JPEGImages',  # 替换为真实的 refer 数据根目录路径
#         ref_file_path='/home/amax/yaoliang/rmsin/refer/data/rrsisd/refs(unc).p',  # 替换为实际的数据集名称
#         instances_path='/home/amax/yaoliang/rmsin/refer/data/rrsisd/instances.json'  # 替换为实际的划分方式
#     )
#     # 定义简单的图像变换，这里仅示例，你可以根据实际需求完善或替换
#     image_transforms = transforms.Compose([
#         transforms.Resize((480, 480)),
#         transforms.ToTensor()
#     ])
#
#     # 实例化 ReferDataset
#     dataset = All_Dataset(
#         args,
#         image_transforms=image_transforms,
#         max_tokens=20,
#         split='val',
#         eval_mode=True
#     )
#
#     print(f"数据集长度: {len(dataset)}")
#
#     # 尝试获取第一个数据样本进行查看
#     if len(dataset) > 0:
#         img, target, tensor_embeddings, attention_mask, exist = dataset[0]
#
#         print("图像数据形状:", img.shape)
#         print("目标数据类型:", type(target))
#         if isinstance(target, torch.Tensor):
#             print("目标数据形状:", target.shape)
#         print("嵌入向量形状:", tensor_embeddings.shape)
#         print("注意力掩码形状:", attention_mask.shape)
#         print("存在性张量形状:", exist.shape)
#
# if __name__ == "__main__":
#     main()