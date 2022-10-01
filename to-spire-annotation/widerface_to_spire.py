
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
    parser = argparse.ArgumentParser(description="Convert WIDER-face annotation to spire annotation")
    parser.add_argument(
        "--widerface-gt-txt",
        default="C:/dataset/WIDERFACE/wider_face_split/wider_face_val_bbx_gt.txt",
        help="path to wider_face_xxx_bbx_gt.txt file",
        # required=True
    )
    parser.add_argument(
        "--widerface-image-dir",
        default="C:/dataset/WIDERFACE/\WIDER_val/images",
        help="path to WIDER_xxx/images",
    )
    parser.add_argument(
        "--output-dir",
        default="C:/dataset/WIDERFACE/BB211123-widerface-val",
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

    f = open(args.widerface_gt_txt, 'r')
    lines = [l.strip() for l in f.readlines()]
    n_lines = len(lines)
    n_imgs = 0
    new_img = False
    l_count = 0
    while l_count < n_lines:
        if lines[l_count].endswith('.jpg'):
            n_imgs += 1
            img_name = os.path.basename(lines[l_count])
            img_fn = os.path.join(args.widerface_image_dir, lines[l_count])
            print("{}, New Image -- [{}]".format(str(n_imgs).zfill(6), img_fn))
            new_img = True
            l_count += 1
            continue
        if new_img:
            n_objs = int(lines[l_count])
            l_count += 1

            # Prepare JSON dictionary for a single image.
            spire_dict = {}
            spire_dict['file_name'] = img_name
            img = cv2.imread(img_fn)
            img_h, img_w = img.shape[0], img.shape[1]
            spire_dict['height'], spire_dict['width'] = img_h, img_w
            spire_dict['annos'] = []

            invalid_image = False
            for j in range(n_objs):
                # print(lines[l_count])
                x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose = \
                    lines[l_count].split(' ')
                x1, y1, w, h = float(x1), float(y1), float(w), float(h)
                blur, expression, illumination, invalid, occlusion, pose = \
                    int(blur), int(expression), int(illumination), int(invalid), int(occlusion), int(pose)

                spire_anno = {}
                spire_anno['area'] = w * h
                spire_anno['bbox'] = [x1, y1, w, h]
                spire_anno['category_name'] = 'face'
                spire_anno['obj_attrs'] = {'clear': 1 if blur == 0 else 0,
                                           'normal_blur': 1 if blur == 1 else 0,
                                           'heavy_blur': 1 if blur == 2 else 0,
                                           'typical_expression': 1 if expression == 0 else 0,
                                           'exaggerate_expression': 1 if expression == 1 else 0,
                                           'normal_illumination': 1 if illumination == 0 else 0,
                                           'extreme_illumination': 1 if illumination == 1 else 0,
                                           'valid_image': 1 if invalid == 0 else 0,
                                           'invalid_image': 1 if invalid == 1 else 0,
                                           'no_occlusion': 1 if occlusion == 0 else 0,
                                           'partial_occlusion': 1 if occlusion == 1 else 0,
                                           'heavy_occlusion': 1 if occlusion == 2 else 0,
                                           'typical_pose': 1 if pose == 0 else 0,
                                           'atypical_pose': 1 if pose == 1 else 0,
                                           }
                if invalid == 1:
                    invalid_image = True
                if w * h > 100:  # 仅保留100像素以上目标
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 255), 1)
                    spire_dict['annos'].append(spire_anno)

                l_count += 1

            new_img = False
            # Generate spire annotation files for each image
            output_fn = os.path.join(args.output_dir, img_name + '.json')
            with open(output_fn, "w") as f:
                json.dump(spire_dict, f)

            if args.save_image:
                open(os.path.join(scaled_images, img_name), 'wb').write(
                    open(img_fn, 'rb').read())

            if args.show_image:
                cv2.imshow('img', img)
                cv2.waitKey(100)
        else:
            l_count += 1


if __name__ == '__main__':
    main()
