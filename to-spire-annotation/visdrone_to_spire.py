
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
    parser = argparse.ArgumentParser(description="Convert VisDrone annotation to spire annotation")
    parser.add_argument(
        "--visdrone-anno",
        default="D:/Dataset/VisDrone2019-DET-val/annotations",
        help="path to visdrone annotation file",
    )
    parser.add_argument(
        "--visdrone-image-dir",
        default="D:/Dataset/VisDrone2019-DET-val/images",
        help="path to visdrone image dir",
    )
    parser.add_argument(
        "--visdrone-meta",
        default="D:/Dataset/VisDrone2019-DET-train/meta_data",
        help="path to visdrone meta_data dir",
    )
    parser.add_argument(
        "--output-dir",
        default="D:/Dataset/BB210426-visdrone19-val",
        help="path to spire home dir",
    )
    args = parser.parse_args()

    scaled_images = os.path.join(args.output_dir, 'scaled_images')
    args.output_dir = os.path.join(args.output_dir, 'annotations')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(scaled_images):
        os.makedirs(scaled_images)

    meta_txt_lst = []
    if os.path.exists(args.visdrone_meta):
        txt_names = os.listdir(args.visdrone_meta)
        for txt in tqdm(txt_names):
            if txt[-4:] == '.txt':
                meta_txt_lst.append(txt)

    category_id_to_name = ['ignored-regions',
                           'pedestrian',
                           'people',
                           'bicycle',
                           'car',
                           'van',
                           'truck',
                           'tricycle',
                           'awning-tricycle',
                           'bus',
                           'motor',
                           'others']

    txt_names = os.listdir(args.visdrone_anno)
    for txt in tqdm(txt_names):
        if txt[-4:] == '.txt':
            image_fn = os.path.join(args.visdrone_image_dir, txt[:-4] + '.jpg')
            image = cv2.imread(image_fn)
            txt_fn = os.path.join(args.visdrone_anno, txt)
            f = open(txt_fn, 'r')
            anno_lines = f.readlines()

            # Prepare JSON dictionary for a single image.
            spire_dict = {}

            if txt in meta_txt_lst:
                print(txt)
                txtm_fn = os.path.join(args.visdrone_meta, txt)
                fm = open(txtm_fn, 'r')
                meta_line = fm.readline()
                print(meta_line)
                daylight, night, low_alt, medium_alt, high_alt, front_view, side_view, bird_view, _ = \
                    meta_line.split(',')
                daylight, night, low_alt, medium_alt, high_alt, front_view, side_view, bird_view = \
                    int(daylight), int(night), int(low_alt), int(medium_alt), int(high_alt), int(front_view), \
                    int(side_view), int(bird_view)
                img_attrs = {"daylight": daylight,
                             "night": night,
                             "fog": 0,
                             "low-alt": low_alt,
                             "medium-alt": medium_alt,
                             "high-alt": high_alt,
                             "front-view": front_view,
                             "side-view": side_view,
                             "bird-view": bird_view}
                spire_dict['img_attrs'] = img_attrs

            spire_dict['file_name'] = txt[:-4] + '.jpg'
            spire_dict['height'], spire_dict['width'] = image.shape[0], image.shape[1]
            spire_dict['annos'] = []
            for line in anno_lines:
                line = line.strip()
                if len(line) > 0:
                    attrs = line.split(sep=',')
                    bbox_left = float(attrs[0])
                    bbox_top = float(attrs[1])
                    bbox_width = float(attrs[2])
                    bbox_height = float(attrs[3])
                    score = float(attrs[4])
                    object_category = int(attrs[5])
                    truncation = int(attrs[6])
                    occlusion = int(attrs[7])
                    spire_anno = {}
                    spire_anno['area'] = bbox_width * bbox_height
                    spire_anno['bbox'] = [bbox_left, bbox_top, bbox_width, bbox_height]
                    spire_anno['truncation'] = truncation
                    spire_anno['occlusion'] = occlusion
                    spire_anno['category_name'] = category_id_to_name[object_category]
                    spire_anno['obj_attrs'] = {'occlusion': occlusion,
                                               'truncation': truncation,
                                               'invisibility': 0}
                    spire_dict['annos'].append(spire_anno)

            # Generate spire annotation files for each image
            output_fn = os.path.join(args.output_dir, txt[:-4] + '.jpg' + '.json')
            with open(output_fn, "w") as f:
                json.dump(spire_dict, f)

            open(os.path.join(scaled_images, txt[:-4] + '.jpg'), 'wb').write(
                open(image_fn, 'rb').read())


if __name__ == '__main__':
    main()
