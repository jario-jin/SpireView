
import argparse
import os
import json
import cv2
from tqdm import tqdm
import pycocotools.mask as maskUtils
import numpy as np
import cv2
from xml.dom.minidom import parse
import xml.dom.minidom


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
    parser = argparse.ArgumentParser(description="IEEE RAL Drone annotation to spire annotation")
    parser.add_argument(
        "--drone-anno",
        default="D:/dataset/IEEE_RAL_Drone/020_annotations",
        help="path to annotation file",
    )
    parser.add_argument(
        "--drone-image-dir",
        default="D:/dataset/IEEE_RAL_Drone/020_images",
        help="path to image dir",
    )
    parser.add_argument(
        "--output-dir",
        default="D:/dataset/IEEE_RAL_Drone/IEEE_RAL_020",
        help="path to spire home dir",
    )
    args = parser.parse_args()

    scaled_images = os.path.join(args.output_dir, 'scaled_images')
    args.output_dir = os.path.join(args.output_dir, 'annotations')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(scaled_images):
        os.makedirs(scaled_images)

    for xml_name in os.listdir(args.drone_anno):
        if xml_name[-4:] == '.xml':
            image_fn = os.path.join(args.drone_image_dir, xml_name[:-4] + '.jpg')
            cv_image = cv2.imread(image_fn)
            xml_fn = os.path.join(args.drone_anno, xml_name)
            # 读取XML文件
            # 使用minidom解析器打开 XML 文档
            DOMTree = xml.dom.minidom.parse(xml_fn)
            annotation = DOMTree.documentElement
            # 文件名
            filename = annotation.getElementsByTagName("filename")[0]
            filename = filename.childNodes[0].data
            if not filename.endswith('.jpg'):
                filename = filename + '.jpg'

            filename = xml_name[:-4] + '.jpg'

            image_size = annotation.getElementsByTagName("size")[0]
            image_width = int(image_size.getElementsByTagName("width")[0].childNodes[0].data)
            image_height = int(image_size.getElementsByTagName("height")[0].childNodes[0].data)
            image_depth = int(image_size.getElementsByTagName("depth")[0].childNodes[0].data)

            w_scale, h_scale = 1., 1.
            if cv_image.shape[1] != image_width or cv_image.shape[0] != image_height:
                print(filename)
                print("image ({}, {}), xml ({}, {})".format(cv_image.shape[1], cv_image.shape[0], image_width, image_height))
                print("---------")
                # w_scale, h_scale = cv_image.shape[1] / float(image_width), cv_image.shape[0] / float(image_height)
                # image_width = cv_image.shape[1]
                # image_height = cv_image.shape[0]
                continue

            # Prepare JSON dictionary for a single image.
            spire_dict = {}
            spire_dict['file_name'] = filename
            spire_dict['height'], spire_dict['width'] = image_height, image_width
            spire_dict['annos'] = []

            # 每个物体
            objects = annotation.getElementsByTagName("object")
            for object_one in objects:
                obj_name = object_one.getElementsByTagName("name")[0].childNodes[0].data
                bnd_box = object_one.getElementsByTagName("bndbox")
                xmin = bnd_box[0].getElementsByTagName("xmin")[0].childNodes[0].data
                ymin = bnd_box[0].getElementsByTagName("ymin")[0].childNodes[0].data
                xmax = bnd_box[0].getElementsByTagName("xmax")[0].childNodes[0].data
                ymax = bnd_box[0].getElementsByTagName("ymax")[0].childNodes[0].data

                pose = object_one.getElementsByTagName("pose")[0].childNodes[0].data
                try:
                    truncated = object_one.getElementsByTagName("truncated")[0].childNodes[0].data
                except IndexError or ValueError:
                    truncated = '0'
                try:
                    difficult = object_one.getElementsByTagName("difficult")[0].childNodes[0].data
                except IndexError or ValueError:
                    difficult = '0'

                xmin, ymin, xmax, ymax = int(round(float(xmin)*w_scale)), int(round(float(ymin)*h_scale)), \
                                         int(round(float(xmax)*w_scale)), int(round(float(ymax)*h_scale))
                bbox_left, bbox_top = xmin, ymin
                bbox_width, bbox_height = xmax - xmin + 1, ymax - ymin + 1
                assert bbox_width > 0 and bbox_height > 0, "ASSERT FALSE: (bbox_width > 0 and bbox_height > 0)"

                spire_anno = {}
                spire_anno['area'] = bbox_width * bbox_height
                spire_anno['bbox'] = [bbox_left, bbox_top, bbox_width, bbox_height]
                spire_anno['pose'] = pose
                spire_anno['truncated'] = int(truncated)
                spire_anno['difficult'] = int(difficult)
                spire_anno['category_name'] = obj_name
                spire_dict['annos'].append(spire_anno)

            # Generate spire annotation files for each image
            output_fn = os.path.join(args.output_dir, filename + '.json')
            with open(output_fn, "w") as f:
                json.dump(spire_dict, f)

            open(os.path.join(scaled_images, filename), 'wb').write(
                open(image_fn, 'rb').read())


if __name__ == '__main__':
    main()
