import torch
import os
from PIL import Image
import json
from tqdm import tqdm
import numpy as np


class UCM_CAP(torch.utils.data.Dataset):

    # potential classes
    classes = ["road", "highway", "bank", "tenniscourt", "roof", "vehicle", "parking lot", "airplane", "stand", "farmland", "waste land", "basketballcourt", "storagetank", "pedestrian", "villa", "beach", "boat", "tree", "grass", "water", "trail", "park", "bush", "golffield", "house", "airport", "people", "overpass", "baseballfield", "swimming-pool", "bunker", "building", "harbor", "residential", "sand", "stadium", "intersection"]

    def __init__(self, args, max_tokens=20) :

        self.args = args
        self.split = args.split
        self.max_tokens = max_tokens

        self.image_path = args.imageFolder
        self.anno_path = args.annoFolder

        self.image_list, self.image2groundtruth, self.image2id = self.load_data()

    
    def load_data(self):

        image_list = []
        image2groundtruth = {}
        image2id = {}

        with open(self.anno_path, 'r') as f:
            data = json.load(f)
        
        for item in tqdm(data['images'], desc="Collecting groundtruth", total=len(data['images'])):
            if item['split'] != self.split:
                continue

            image_name = item["filename"]
            if image_name not in image2groundtruth:
                image2groundtruth[image_name] = []
                image_list.append(image_name)
                image2id[image_name] = item["imgid"]

            for gt in item["sentences"]:
                caption = gt["raw"]
                image2groundtruth[image_name].append(caption)

        return sorted(image_list), image2groundtruth, image2id


    def __getitem__(self, idx):

        image_name = self.image_list[idx]
        origin_image = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')

        # return image, gt, image_name
        return torch.tensor(np.array(origin_image)), self.image2groundtruth[image_name], image_name


    def __len__(self):
        return len(self.image_list)