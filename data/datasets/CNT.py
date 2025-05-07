import torch
import os
from PIL import Image
import json
from tqdm import tqdm
import numpy as np


class Falcon_CNT(torch.utils.data.Dataset):

    convert_name = None
    prior_area_info = None

    def __init__(self, args, max_tokens=20) :

        self.args = args
        self.split = args.split
        self.max_tokens = max_tokens

        self.image_path = args.imageFolder
        self.anno_path = args.annoFolder

        self.image_list, self.groundtruth, self.classnames, self.foo = self.load_data()


    def load_data(self):

        image_list = []
        groundtruth = []
        classnames = []
        foo = []

        with open(self.anno_path, 'r') as f:
            data = json.load(f)
            
        for item in tqdm(data, desc="Generating sentence embeddings", total=len(data)):
            classname = item["conversations"][0]["content"].split("the number of")[-1].split(".")[0].strip()
            if classname in self.convert_name:
                classname = self.convert_name[classname]

            classnames.append(classname)

            image_list.append(os.path.basename(item["images"][0]))
            groundtruth.append(int(item["conversations"][1]["content"]))

            if self.prior_area_info is not None:
                foo.append(self.prior_area_info[classname])
            else:
                foo.append(None)

        return image_list, groundtruth, classnames, foo

            
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        origin_image = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')

        # return image, gt, image_name, classname, foo
        return torch.tensor(np.array(origin_image)), self.groundtruth[idx], image_name, self.classnames[idx], self.foo[idx] 


    def __len__(self):
        return len(self.image_list)


class DIOR_CNT(Falcon_CNT):

    # unify classnames
    convert_name = {
        'railway station':          'trainstation',
        'baseball field':           'baseballfield', 
        'basketball court':         'basketballcourt', 
        'tennis court':             'tenniscourt', 
        'ground track field':       'groundtrackfield', 
        'expressway toll station':  'expressway-toll-station', 
        'storage tank':             'storagetank', 
        'golf field':               'golffield', 
        'expressway service area':  'expressway-service-area'
    }   

    prior_area_info = {'airplane': 36, 'airport': 1824, 'baseballfield': 20, 'basketballcourt': 88, 'bridge': 9, 'chimney': 12, 'dam': 264, 'expressway-service-area': 110, 'expressway-toll-station': 36, 'golffield': 2728, 'groundtrackfield': 132, 'harbor': 24, 'overpass': 9, 'ship': 14, 'stadium': 255, 'storagetank': 4, 'tenniscourt': 36, 'trainstation': 1170, 'vehicle': 6, 'windmill': 27}


class DOTA_CNT(Falcon_CNT):

    # unify classnames
    convert_name = {
        "baseball field":           "baseballfield",
        "basketball court":         "basketballcourt",
        "crane":                    "container-crane",
        "ground track field":       "groundtrackfield",
        "soccer ball field":        "soccer-ball-field",
        "storage tank":             "storagetank",
        "swimming pool":            "swimming-pool",
        "tennis court":             "tenniscourt",
    }

    prior_area_info = {'airplane': 35, 'ship': 15, 'storagetank': 8, 'baseballfield': 81, 'tenniscourt': 153, 'basketballcourt': 576, 'groundtrackfield': 210, 'harbor': 64, 'bridge': 24, 'vehicle': 21, 'helicopter': 162, 'roundabout': 16, 'soccer-ball-field': 72, 'swimming-pool': 20, "container-crane": 187, "airport": 1337, "helipad":121}