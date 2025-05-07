import os
import random
import xml.etree.ElementTree as ET
import json
from .metric.cidereval.eval import CIDErEvalCap
from pycocotools.coco import COCO

def object_existence_cap(objects):
    sentence_templates = [
        "It contains {}.",
        "The image includes {}.",
        "There are {} in the image.",
        "This image features {}.",
        "This picture shows {}."
    ]

    obj_count = {}
    for obj in objects:
        obj_count[obj] = obj_count.get(obj, 0) + 1

    object_names = []
    for obj, count in obj_count.items():
        if count >= 5:
            object_names.append(f"many {obj}s")
        else:
            object_names.append(f"{count} {obj}s" if count > 1 else obj)

    # random select a sentence template
    sentence_template = random.choice(sentence_templates)

    return sentence_template.format(", ".join(object_names))


def object_distribution_cap(objects, width, height, region_count):
    assert region_count in [4, 9], "region_count must be 4 or 9"

    split = int(region_count ** 0.5)

    # 0[0]: object names, 0[1]: region, 0[2]: is/are
    sentence_templates = [
        "The {0[1]} region contains {0[0]}.",
        "{0[0]} {0[2]} distributed in the {0[1]}.",
        "There {0[2]} {0[0]} in the {0[1]}.",
        "The {0[1]} region has {0[0]}.",
        "{0[0]} {0[2]} observed in the {0[1]}.",
        "In the {0[1]}, {0[0]} {0[2]} present.",
    ]

    region_width = width / split
    region_height = height / split

    if region_count == 9:

        # ┌──────────────┬───────────────┬───────────────┐
        # │  upper left  │  upper center │  upper right  │
        # ├──────────────┼───────────────┼───────────────┤
        # │  middle left │    center     │  middle right │
        # ├──────────────┼───────────────┼───────────────┤
        # │  lower left  │  lower center │  lower right  │
        # └──────────────┴───────────────┴───────────────┘

        regions = [
            ["upper left", "upper center", "upper right"],
            ["middle left", "center", "middle right"],
            ["lower left", "lower center", "lower right"],
        ]

    elif region_count == 4:

        regions = [
            ["upper left", "upper right"],
            ["lower left", "lower right"],
        ]

    # fit into regions based on center coordinates
    region_objects = {}

    for i in range(split):
        for j in range(split):
            region_objects[regions[i][j]] = {}
    
    for obj, coords in objects:
        center_x = (coords[0] + coords[2]) / 2
        center_y = (coords[1] + coords[3]) / 2

        loc_i = int(center_y // region_height)
        loc_j = int(center_x // region_width)

        region = regions[loc_i][loc_j]

        if obj not in region_objects[region]:
            region_objects[region][obj] = 0
        region_objects[region][obj] += 1

    distribution_sentences = []
    for region, objs in region_objects.items():
        if objs:
            components = ["", region, ""]
            cum_cnt = 0 
            for obj, cnt in objs.items():
                cum_cnt += cnt
                if cnt > 1:
                    components[0] += f"{obj}s, "
                else:
                    components[0] += f"{obj}, "
            
            components[2] = "are" if cum_cnt > 1 else "is"
            components[0] = components[0][:-2]

            # random select a sentence template
            sentence_template = random.choice(sentence_templates)
            distribution_sentences.append(sentence_template.format(components))

    return " ".join(distribution_sentences)


def merge_json(jsonpath):
    """
    Merge multiple json files from Object Detection into one list
    """
    filelist = [file for file in os.listdir(jsonpath) if file.endswith('.json')]
    result = []
    for file in filelist:
        with open(os.path.join(jsonpath, file), 'r') as f:
            data = json.load(f)
            result += data
    result.sort(key=lambda x: x['image_id'])
    return result


def evaluate(gtjson, predjson):
    cocoGt = COCO(gtjson)
    cocoDt = cocoGt.loadRes(predjson)
    cocoeval_cap = CIDErEvalCap(cocoGt, cocoDt)
    cocoeval_cap.evaluate()

    print("\n########## Evaluation Summary ##########")
    print("CIDEr:\t{:.3f}%".format(cocoeval_cap.eval['CIDEr'] * 100.))
    print()


def single_captioning(pred, shape, region_count):

    assert isinstance(pred, dict)
    
    objects = []
    for classname in pred:
        for box in pred[classname]:
            objects.append((classname, box[:4]))

    # part1: object existence
    object_names = [obj[0] for obj in objects]
    sentence_1 = object_existence_cap(object_names)

    # part2: object distribution
    sentence_2 = object_distribution_cap(objects, shape[1], shape[0], region_count=region_count)

    return sentence_1 + " " + sentence_2


def captioning(gtjson, predjson, output_json):

    image_captions = []
    all_pred = merge_json(predjson)

    image2pred = {}
    for pred in all_pred:
        if pred['image_id'] not in image2pred:
            image2pred[pred['image_id']] = []
        image2pred[pred['image_id']].append(pred)

    with open(gtjson, 'r') as f:
        gtdata = json.load(f)

    for item in gtdata['images']:
        imageid = item['id']

        if imageid not in image2pred:
            # form empty caption for json
            image_captions.append({
                "image_id": imageid,
                "caption": ""
            })
            continue

        objects = []
        for obj in image2pred[imageid]:
            box = obj['bbox']
            box[2] += box[0]
            box[3] += box[1]
            objects.append((obj['category_id'], box))

        # part1: object existence
        object_names = [obj[0] for obj in objects]
        sentence_1 = object_existence_cap(object_names)

        # part2: object distribution
        sentence_2 = object_distribution_cap(objects, item['width'], item['height'], region_count=4)

        # form json
        image_captions.append({
            "image_id": imageid,
            "caption": sentence_1 + " " + sentence_2
        })

    with open(output_json, 'w') as f:
        json.dump(image_captions, f, indent=4)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--gt_json', type=str, default='', help='Path to the ground truth JSON file')
    parser.add_argument('--detections', type=str, default='', help='Path to the folder of detection JSON files')
    parser.add_argument('--output_json', type=str, default='', help='Path to the output JSON file')
    parser.add_argument('--eval', action='store_true', help='Whether to evaluate the output JSON file')
    args = parser.parse_args()

    captioning(args.gt_json, args.detections, args.output_json)

    if args.eval:
        evaluate(args.gt_json, args.output_json)