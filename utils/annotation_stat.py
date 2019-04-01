
import argparse
import os
import json
import cv2
import numpy as np


def open_spire_annotations(spire_dir):
    if os.path.exists(os.path.join(spire_dir, 'annotations')):
        spire_dir = os.path.join(spire_dir, 'annotations')

    image_jsons = []
    for f in os.listdir(spire_dir):
        if f.endswith('.json'):
            json_f = open(os.path.join(spire_dir, f), 'r')
            json_str = json_f.read()
            json_dict = json.loads(json_str)
            image_jsons.append(json_dict)
    return image_jsons


def main():
    parser = argparse.ArgumentParser(description="Statistic on spire annotations")
    parser.add_argument(
        "--spire-anno",
        default="/tmp/coco_spire",
        help="path to spire annotation dir",
        required=True
    )
    args = parser.parse_args()

    image_jsons = open_spire_annotations(args.spire_anno)
    image_width_min, image_height_min = np.inf, np.inf
    image_width_max, image_height_max = 0, 0
    bbox_width_min, bbox_height_min = np.inf, np.inf
    bbox_width_max, bbox_height_max = 0, 0
    area_min, area_max = np.inf, 0
    boxnum_each_image_min, boxnum_each_image_max = np.inf, 0

    for image_anno in image_jsons:
        boxnum_each_image_min = np.minimum(boxnum_each_image_min, len(image_anno['annos']))
        boxnum_each_image_max = np.maximum(boxnum_each_image_max, len(image_anno['annos']))
        image_width_min = np.minimum(image_width_min, image_anno['width'])
        image_width_max = np.maximum(image_width_max, image_anno['width'])
        image_height_min = np.minimum(image_height_min, image_anno['height'])
        image_height_max = np.maximum(image_height_max, image_anno['height'])

        for anno in image_anno['annos']:
            bbox = anno['bbox']
            bbox_width, bbox_height = bbox[2], bbox[3]
            bbox_width_min = np.minimum(bbox_width_min, bbox_width)
            bbox_width_max = np.maximum(bbox_width_max, bbox_width)
            bbox_height_min = np.minimum(bbox_height_min, bbox_height)
            bbox_height_max = np.maximum(bbox_height_max, bbox_height)
            area_min = np.minimum(area_min, anno['area'])
            area_max = np.maximum(area_max, anno['area'])

    print('image_width_min:{:0.2f}, image_height_min:{:0.2f}'.format(image_width_min, image_height_min))
    print('image_width_max:{:0.2f}, image_height_max:{:0.2f}'.format(image_width_max, image_height_max))

    print('bbox_width_min:{:0.2f}, bbox_height_min:{:0.2f}'.format(bbox_width_min, bbox_height_min))
    print('bbox_width_max:{:0.2f}, bbox_height_max:{:0.2f}'.format(bbox_width_max, bbox_height_max))

    print('area_min:{:0.2f}, area_max:{:0.2f}'.format(area_min, area_max))
    print('boxnum_each_image_min:{:0.2f}, boxnum_each_image_max:{:0.2f}'.format(boxnum_each_image_min, boxnum_each_image_max))


if __name__ == '__main__':
    main()
