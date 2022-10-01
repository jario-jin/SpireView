
import argparse
import os
import json
import cv2
import numpy as np
from annotation_stat import open_spire_annotations


def main():
    parser = argparse.ArgumentParser(description="Statistic on spire annotations")
    parser.add_argument(
        "--spire-annos",
        default="/home/jario/dataset/BB200326_mbzirc_c1_yellow_val/annotations",
        help="path to spire annotation dir",
    )
    parser.add_argument(
        "--spire-images",
        default="/home/jario/dataset/BB200326_mbzirc_c1_yellow_val/scaled_images",
        help="path to spire image dir",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/jario/dataset/BB200326_mbzirc_c1_yellow_train_ROI",
        help="path to output dir",
    )
    parser.add_argument(
        "--roi-min",
        default=32,
        help="retain the boxes that area >= 'area-min'",
    )
    parser.add_argument(
        "--roi-max",
        default=600,
        help="retain the boxes that area <= 'area-max'",
    )
    args = parser.parse_args()
    image_jsons = open_spire_annotations(args.spire_annos)

    if os.path.exists(os.path.join(args.spire_annos, 'annotations')):
        args.spire_anno = os.path.join(args.spire_annos, 'annotations')

    roi_cnt = 1
    for i, image_anno in enumerate(image_jsons):
        file_name = image_anno['file_name']
        h, w = image_anno['height'], image_anno['width']
        image = cv2.imread(os.path.join(args.spire_images, file_name))
        assert image.shape[0] == h and image.shape[1] == w, 'miss align of image and json img_shape'

        print('[{} of {}]'.format(i, len(image_jsons)))
        for anno in image_anno['annos']:
            x, y, w, h = anno['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            x = max(0, x)
            y = max(0, y)
            cat = anno['category_name']

            if w < args.roi_min or h < args.roi_min:
                continue

            roi = image[y:y + h, x:x + w]
            # print((x, y, w, h))
            if w > args.roi_max or h > args.roi_max:
                img_resize = args.roi_max
                if h < w:
                    h = int(float(h) / w * img_resize)
                    w = img_resize
                else:
                    w = int(float(w) / h * img_resize)
                    h = img_resize
                # print("roi: {}, re: {}".format(roi.shape, (w, h)))
                roi = cv2.resize(roi, (w, h))

            output_path = os.path.join(args.output_dir, cat)
            if not os.path.exists(output_path):
                os.mkdir(output_path)

            name = "%06d.jpg" % roi_cnt
            roi_cnt += 1
            cv2.imwrite(os.path.join(output_path, name), roi)


if __name__ == '__main__':
    main()
