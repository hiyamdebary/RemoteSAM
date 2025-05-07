import torch
import os
from PIL import Image
import json
from tqdm import tqdm
import numpy as np


class COCO_style(torch.utils.data.Dataset):

    classes = None

    def __init__(self, args, max_tokens=20) :

        assert args._class in self.classes, "class not in the list of classes"

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
        
        id2image = {}
        for item in data['images']:
            id2image[item["id"]] = item["file_name"]

        for item in tqdm(data['annotations'], desc="Collecting groundtruth", total=len(data['annotations'])):
            image_name = id2image[item["image_id"]]
            if image_name not in image2groundtruth:
                image2groundtruth[image_name] = []
                image_list.append(image_name)
                image2id[image_name] = item["image_id"]

            if item["category_id"] == self.classes.index(self.args._class):
                x, y, w, h = item["bbox"]
                box = [x, y, x+w, y+h]
                image2groundtruth[image_name].append(box)

        return sorted(image_list), image2groundtruth, image2id

            
    def __getitem__(self, idx):

        image_name = self.image_list[idx]
        origin_image = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')
        
        # return image, gt, image_name
        return torch.tensor(np.array(origin_image)), torch.tensor(self.image2groundtruth[image_name]), image_name


    def __len__(self):
        return len(self.image_list)


class DOTA_style(torch.utils.data.Dataset):

    classes = None
    convert_name = None

    def __init__(self, args, max_tokens=20) :

        assert args._class in self.classes, "class not in the list of classes"

        self.args = args
        self.split = args.split
        self.max_tokens = max_tokens

        self.image_path = args.imageFolder
        self.anno_path = args.annoFolder

        self.image_list, self.image2groundtruth = self.load_data()


    def load_data(self):

        image_list = os.listdir(self.image_path)
        image2groundtruth = {}

        for image_name in image_list:
            image2groundtruth[image_name] = []

        # skip for no gt available in DOTA test set
        if self.split != 'test':
            for image_name in tqdm(image_list, desc="Collecting groundtruth", total=len(image_list)):
                anno_name = os.path.splitext(image_name)[0] + ".txt"
                with open(os.path.join(self.anno_path, anno_name), 'r') as f:
                    for line in f:
                        data = line.strip().split(' ')
                        if data[8] in self.convert_name:
                            data[8] = self.convert_name[data[8]]
                        if data[8] == self.args._class:
                            image2groundtruth[image_name].append([int(float(a)) for a in data[:8]])

        return sorted(image_list), image2groundtruth


    def __getitem__(self, idx):

        image_name = self.image_list[idx]
        origin_img = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')

        # return image, gt, image_name
        return torch.tensor(np.array(origin_img)), torch.tensor(self.image2groundtruth[image_name]), image_name


    def __len__(self):
        return len(self.image_list)
    

class DIOR_DET(COCO_style):

    classes = ['background', 'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam', 'expressway-service-area', 'expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill']


class DOTAv2_DET(DOTA_style):

    classes = ['airplane', 'ship', 'storagetank', 'baseballfield', 'tenniscourt', 'basketballcourt', 'groundtrackfield', 'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool', "container-crane", "airport", "helipad"]

    # unify classname
    convert_name = {
        'plane': 'airplane',
        'storage-tank': 'storagetank',
        'baseball-diamond': 'baseballfield',
        'tennis-court': 'tenniscourt',
        'basketball-court': 'basketballcourt',
        'ground-track-field': 'groundtrackfield',
    }


class DOTAv1_DET(DOTA_style):

    classes = ['plane', 'ship', 'storage-tank', 'baseball-datdmond', 'tennis-court', 'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool']

    # unify classname
    convert_name = {
        'plane': 'airplane',
        'storage-tank': 'storagetank',
        'baseball-diamond': 'baseballfield',
        'tennis-court': 'tenniscourt',
        'basketball-court': 'basketballcourt',
        'ground-track-field': 'groundtrackfield',
    }