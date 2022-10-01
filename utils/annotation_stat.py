
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


def topdown_percent_wh(bboxes, percent=0.2):
    """
    :param bboxes: np.float32, [n, 4], x,y,w,h
    :param percent: (0, 1]
    :return: min_w, max_w, min_h, max_h
    """
    if bboxes.ndim == 1:
        bboxes = np.expand_dims(bboxes, axis=0)

    num_samp = int(np.ceil(len(bboxes) * percent))
    width_ids = np.argsort(bboxes[:, 2])
    height_ids = np.argsort(bboxes[:, 3])
    min_w = bboxes[width_ids[:num_samp], 2]
    min_w = np.mean(min_w)
    max_w = bboxes[width_ids[-num_samp:], 2]
    max_w = np.mean(max_w)
    min_h = bboxes[height_ids[:num_samp], 3]
    min_h = np.mean(min_h)
    max_h = bboxes[height_ids[-num_samp:], 3]
    max_h = np.mean(max_h)
    return min_w, max_w, min_h, max_h


def main():
    parser = argparse.ArgumentParser(description="Statistic on spire annotations")
    parser.add_argument(
        "--spire-anno",
        default="D:\\Dataset\\zhuangbei",
        help="path to spire annotation dir",
        # required=True
    )
    args = parser.parse_args()
    use_topdown_percent = True

    image_jsons = open_spire_annotations(args.spire_anno)
    image_width_min, image_height_min = np.inf, np.inf
    image_width_max, image_height_max = 0, 0
    bbox_width_min, bbox_height_min = np.inf, np.inf
    bbox_width_max, bbox_height_max = 0, 0
    area_min, area_max = np.inf, 0
    boxnum_each_image_min, boxnum_each_image_max = np.inf, 0

    for image_anno in image_jsons:
        width, height = int(image_anno['width']), int(image_anno['height'])
        file_name = image_anno['file_name']
        image_dir = os.path.join(args.spire_anno, 'scaled_images')
        image_fn = os.path.join(image_dir, file_name)
        image = cv2.imread(image_fn)
        if width != image.shape[1] or height != image.shape[0]:
            print('ERROR: {}'.format(file_name))

        boxnum_each_image_min = np.minimum(boxnum_each_image_min, len(image_anno['annos']))
        boxnum_each_image_max = np.maximum(boxnum_each_image_max, len(image_anno['annos']))
        image_width_min = np.minimum(image_width_min, image_anno['width'])
        image_width_max = np.maximum(image_width_max, image_anno['width'])
        image_height_min = np.minimum(image_height_min, image_anno['height'])
        image_height_max = np.maximum(image_height_max, image_anno['height'])

        bboxes = []
        for anno in image_anno['annos']:
            bbox = np.array(anno['bbox'], np.float32)
            bboxes.append(bbox)
            bbox_width, bbox_height = bbox[2], bbox[3]
            if not use_topdown_percent:
                bbox_width_min = np.minimum(bbox_width_min, bbox_width)
                bbox_width_max = np.maximum(bbox_width_max, bbox_width)
                bbox_height_min = np.minimum(bbox_height_min, bbox_height)
                bbox_height_max = np.maximum(bbox_height_max, bbox_height)
            area_min = np.minimum(area_min, anno['area'])
            area_max = np.maximum(area_max, anno['area'])

        # need at least one array to concatenate
        if len(bboxes) > 0:
            bboxes = np.vstack(bboxes)
            min_w, max_w, min_h, max_h = topdown_percent_wh(bboxes)

            if use_topdown_percent:
                bbox_width_min = np.minimum(bbox_width_min, min_w)
                bbox_width_max = np.maximum(bbox_width_max, max_w)
                bbox_height_min = np.minimum(bbox_height_min, min_h)
                bbox_height_max = np.maximum(bbox_height_max, max_h)

    print('image_width_min:{:0.2f}, image_height_min:{:0.2f}'.format(image_width_min, image_height_min))
    print('image_width_max:{:0.2f}, image_height_max:{:0.2f}'.format(image_width_max, image_height_max))

    print('bbox_width_min:{:0.2f}, bbox_height_min:{:0.2f}'.format(bbox_width_min, bbox_height_min))
    print('bbox_width_max:{:0.2f}, bbox_height_max:{:0.2f}'.format(bbox_width_max, bbox_height_max))

    print('area_min:{:0.2f}, area_max:{:0.2f}'.format(area_min, area_max))
    print('boxnum_each_image_min:{:0.2f}, boxnum_each_image_max:{:0.2f}'.format(boxnum_each_image_min, boxnum_each_image_max))


if __name__ == '__main__':
    main()
