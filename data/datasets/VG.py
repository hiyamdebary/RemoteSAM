import torch
import os
from PIL import Image
import json
from tqdm import tqdm
import xml.etree.cElementTree as ET
import numpy as np


class RSVG_VG(torch.utils.data.Dataset):

    def __init__(self, args, max_tokens=20):

        self.args = args
        self.split = args.split
        self.max_tokens = max_tokens
        
        self.image_path = args.imageFolder
        self.anno_path = args.annoFolder

        self.image_list, self.groundtruth, self.sentence = self.load_data()
    

    def load_data(self):

        image_list = []
        groundtruth = []
        sentence = []
        
        split_txt = os.path.join(os.path.dirname(self.anno_path), self.split + ".txt")
        with open(split_txt, "r", encoding="utf-8") as f:
            lines = f.readlines()
            Index = [ int(line.strip()) for line in lines ]
        
        count = 0
        anno_list = sorted(os.listdir(self.anno_path))
        for anno in tqdm(anno_list, desc="Generating sentence embeddings", total=len(anno_list)):
            root = ET.parse(os.path.join(self.anno_path, anno)).getroot()
            image_name = root.find('filename').text
            objects = []
            for obj in root.findall('object'):
                if count in Index:

                    bndbox = obj.find('bndbox')
                    xmin = bndbox.find('xmin').text
                    ymin = bndbox.find('ymin').text
                    xmax = bndbox.find('xmax').text
                    ymax = bndbox.find('ymax').text
                    box = [int(xmin), int(ymin), int(xmax), int(ymax)]

                    description = obj.find('description').text

                    image_list.append(image_name)
                    groundtruth.append(box)
                    sentence.append(description)
                
                count += 1

        return image_list, groundtruth, sentence


    def __getitem__(self, idx):

        image_name = self.image_list[idx]
        origin_image = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')

        # return image, gt, image_name, sentence
        return torch.tensor(np.array(origin_image)), torch.tensor(self.groundtruth[idx]), image_name, self.sentence[idx]


    def __len__(self):
        return len(self.image_list)