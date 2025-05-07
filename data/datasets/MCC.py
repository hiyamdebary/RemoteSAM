import torch
import os
from PIL import Image
import json
from tqdm import tqdm
import numpy as np


class MCC(torch.utils.data.Dataset):

    classes = None
    convert_name = None

    def __init__(self, args, max_tokens=20) :

        self.args = args
        self.split = args.split
        self.max_tokens = max_tokens

        self.image_path = args.imageFolder
        self.anno_path = args.annoFolder

        self.image_list, self.image2groundtruth = self.load_data()


    def load_data(self):
        image_list = []
        image2groundtruth = {}

        for index, classname in enumerate(self.classes):
            if classname in self.convert_name:
                classname = self.convert_name[classname]
            for image in os.listdir(os.path.join(self.anno_path, classname)):
                image_path = os.path.join(classname, image)
                image2groundtruth[image_path] = index
                image_list.append(image_path)

        return image_list, image2groundtruth

            
    def __getitem__(self, idx):

        image_name = self.image_list[idx]
        origin_image = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')
        
        # return image, gt, image_name
        return torch.tensor(np.array(origin_image)), torch.tensor(self.image2groundtruth[image_name]), image_name


    def __len__(self):
        return len(self.image_list)


class UCM_MCC(MCC):

    classes = ['agricultural', 'airplane', 'baseballfield', 'beach', 'buildings', 'chaparral', 'denseresidential', 'forest', 'freeway', 'golffield', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetank', 'tenniscourt']

    convert_name = {
        'baseballfield': 'baseballdiamond',
        'golffield': 'golfcourse',
        'storagetank': 'storagetanks'
    }


class AID_MCC(MCC):

    classes = ['airport', 'bareland', 'baseballfield', 'beach', 'bridge', 'center', 'church', 'commercial', 'denseresidential', 'desert', 'farmland', 'forest', 'industrial', 'meadow', 'mediumresidential', 'mountain', 'park', 'parking', 'playground', 'pond', 'port', 'railwaystation', 'resort', 'river', 'school', 'sparseresidential', 'square', 'stadium', 'storagetank', 'viaduct']

    convert_name = {
        'storagetank': 'storagetanks'
    }