
import argparse
import os
import json
import cv2
from tqdm import tqdm
import pycocotools.mask as maskUtils
import numpy as np
import cv2


def find_contours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4
    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy


def solve_coco_segs(segs_anno, h, w):
    assert type(segs_anno['counts']) == list, "segs_anno['counts'] should be list"
    rle = maskUtils.frPyObjects(segs_anno, h, w)
    mask = maskUtils.decode(rle)
    contours, hierarchy = find_contours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    for contour in contours:
        area += int(round(cv2.contourArea(contour)))
    if area != 0:
        segmentations = []
        for contour in contours:
            if contour.shape[0] >= 6:  # three points
                segmentation = []
                for cp in range(contour.shape[0]):
                    segmentation.append(int(contour[cp, 0, 0]))
                    segmentation.append(int(contour[cp, 0, 1]))
                segmentations.append(segmentation)
    else:
        segmentations = []
    return segmentations


def main():
    parser = argparse.ArgumentParser(description="Convert COCO annotation to spire annotation")
    parser.add_argument(
        "--open-images-anno",
        default="/media/jario/PLAN_B/open_images/boxes/train-annotations-bbox.csv",
        help="path to open-images annotation file",
        # required=True
    )
    parser.add_argument(
        "--open-images-meta",
        default="/media/jario/PLAN_B/open_images/metadata/class-descriptions-boxable.csv",
        help="path to open-images meta file",
        # required=True
    )
    parser.add_argument(
        "--open-images-dir",
        default="/media/jario/PLAN_B/open_images",
        help="path to open-images image dir",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/open_images_spire",
        help="path to spire home dir",
    )
    parser.add_argument(
        '--show-image',
        action='store_true',
        help='show image for testing'
    )
    args = parser.parse_args()

    scaled_images = os.path.join(args.output_dir, 'scaled_images')
    args.output_dir = os.path.join(args.output_dir, 'annotations')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(scaled_images):
        os.makedirs(scaled_images)

    f = open(args.open_images_meta, 'r')
    cat_strs = f.readlines()
    cat_dict = {}
    for cat_str in cat_strs:
        cat_d, cat_n = cat_str.split(',')[0].strip(), cat_str.split(',')[1].strip()
        if 'Human' in cat_n:
            cat_dict[cat_d] = cat_n

    sub_dirs = ['train_0', 'train_1', 'train_2', 'train_3', 'train_4', 'train_5', 'train_6', 'train_7',
                'train_8', 'train_9', 'train_a', 'train_b', 'train_c', 'train_d', 'train_e', 'train_f']
    img_fns = []
    for sub_dir in sub_dirs:
        fn = os.path.join(args.open_images_dir, sub_dir, sub_dir)
        for img_name in os.listdir(fn):
            if img_name.endswith('.jpg'):
                img_fn = os.path.join(fn, img_name)
                img_fns.append(img_fn)

    f = open(args.open_images_anno, 'r')
    bbox_strs = f.readlines()
    img_bbox_dict = {}
    for bbox_str in tqdm(bbox_strs):
        bbox_items = bbox_str.split(',')
        ImageID, Source, LabelName, Confidence, XMin, XMax, YMin, YMax, IsOccluded, IsTruncated, IsGroupOf, \
        IsDepiction, IsInside = bbox_items[0], bbox_items[1], bbox_items[2], bbox_items[3], bbox_items[4], \
                                bbox_items[5], bbox_items[6], bbox_items[7], bbox_items[8], bbox_items[9], \
                                bbox_items[10], bbox_items[11], bbox_items[12]
        if LabelName in cat_dict.keys():
            # Human
            # print(cat_dict[LabelName])
            if ImageID not in img_bbox_dict.keys():
                img_bbox_dict[ImageID] = []
            img_bbox_dict[ImageID].append([ImageID, Source, LabelName, Confidence, XMin, XMax, YMin, YMax,
                                           IsOccluded, IsTruncated, IsGroupOf, IsDepiction, IsInside])

    for img_name in tqdm(img_bbox_dict.keys()):
        for img_fn in img_fns:
            if img_name in img_fn:
                img = cv2.imread(img_fn)
                img_h, img_w = img.shape[0], img.shape[1]
                # Show image for testing.
                if args.show_image:
                    cv2.imshow('img', img)
                    cv2.waitKey(5)
                # Prepare JSON dictionary for a single image.
                spire_dict = {}
                spire_dict['file_name'] = os.path.basename(img_fn)
                spire_dict['height'], spire_dict['width'] = img_h, img_w
                spire_dict['annos'] = []
                for anno in img_bbox_dict[img_name]:
                    spire_anno = {}
                    x1, y1, x2, y2 = float(anno[4]) * img_w, float(anno[6]) * img_h, \
                                     float(anno[5]) * img_w, float(anno[7]) * img_h
                    x, y, w, h = x1, y1, x2 - x1 + 1, y2 - y1 + 1
                    spire_anno['area'] = w * h
                    spire_anno['bbox'] = [x, y, w, h]
                    spire_anno['iscrowd'] = 0
                    spire_anno['confidence'] = float(anno[3])
                    spire_anno['isoccluded'] = int(anno[8])
                    spire_anno['istruncated'] = int(anno[9])
                    spire_anno['isgroupof'] = int(anno[10])
                    spire_anno['isdepiction'] = int(anno[11])
                    spire_anno['isinside'] = int(anno[12])
                    spire_anno['category_name'] = cat_dict[anno[2]].lower().replace(' ', '_')
                    spire_dict['annos'].append(spire_anno)
                # Generate spire annotation files for each image
                output_fn = os.path.join(args.output_dir, os.path.basename(img_fn) + '.json')
                with open(output_fn, "w") as f:
                    json.dump(spire_dict, f)
                open(os.path.join(scaled_images, os.path.basename(img_fn)), 'wb').write(
                    open(img_fn, 'rb').read())
                break


if __name__ == '__main__':
    main()
