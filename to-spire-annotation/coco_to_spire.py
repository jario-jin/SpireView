
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
        "--coco-anno",
        default="/home/jario/dataset/coco/annotations/instances_val2014.json",
        help="path to coco annotation file",
        required=True
    )
    parser.add_argument(
        "--coco-image-dir",
        default="/home/jario/dataset/coco/val2014",
        help="path to coco image dir",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/coco_spire",
        help="path to spire home dir",
    )
    parser.add_argument(
        '--save-image',
        action='store_true',
        help='save image to scaled_images'
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

    f = open(args.coco_anno, 'r')
    json_str = f.read()
    json_dict = json.loads(json_str)

    # Pack all coco categories to a dictionary.
    category_id_to_name = {}
    for cat in json_dict['categories']:
        category_id_to_name[cat['id']] = cat['name']

    for img_info in tqdm(json_dict['images']):
        img_id = img_info['id']
        img_name = img_info['file_name']
        img_h, img_w = img_info['height'], img_info['width']
        # Find all annotations based on image_id. -> img_annos
        img_annos = []
        for anno in json_dict['annotations']:
            if anno['image_id'] == img_id:
                img_annos.append(anno)

        # Prepare JSON dictionary for a single image.
        spire_dict = {}
        spire_dict['file_name'] = img_name
        spire_dict['height'], spire_dict['width'] = img_h, img_w
        spire_dict['annos'] = []
        for anno in img_annos:
            spire_anno = {}
            spire_anno['area'] = anno['area']
            spire_anno['bbox'] = anno['bbox']
            if isinstance(anno['segmentation'], dict):
                spire_anno['segmentation'] = solve_coco_segs(anno['segmentation'], img_h, img_w)
                assert anno['segmentation']['size'] == [img_h, img_w], "segmentation.size != [img_h, img_w]"
            else:
                spire_anno['segmentation'] = anno['segmentation']
            spire_anno['iscrowd'] = anno['iscrowd']
            category_id = anno['category_id']
            spire_anno['category_name'] = category_id_to_name[category_id]
            if len(spire_anno['segmentation']) > 0:
                spire_dict['annos'].append(spire_anno)
            else:
                print("Image {} no segmentation.".format(img_name))

        # Generate spire annotation files for each image
        output_fn = os.path.join(args.output_dir, img_name+'.json')
        with open(output_fn, "w") as f:
            json.dump(spire_dict, f)

        if args.save_image:
            open(os.path.join(scaled_images, img_name), 'wb').write(
                open(os.path.join(args.coco_image_dir, img_name), 'rb').read())

        # Show image for testing.
        if args.show_image:
            img_fn = os.path.join(args.coco_image_dir, img_name)
            img = cv2.imread(img_fn)
            cv2.imshow('img', img)
            cv2.waitKey(100)
            print(img_name)


if __name__ == '__main__':
    main()
