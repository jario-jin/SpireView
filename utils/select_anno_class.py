
import argparse
import os
import json
import cv2
import numpy as np
from annotation_stat import open_spire_annotations


def main():
    parser = argparse.ArgumentParser(description="Statistic on spire annotations")
    parser.add_argument(
        "--spire-anno",
        default="/home/jario/car-landing-data/200m/annotations",
        help="path to spire annotation dir",
        # required=True
    )
    parser.add_argument(
        "--classes",
        default=('car',),
        help="selected class ids",
        # required=True
    )
    SAVING_ROI = False
    SAVING_SPIRE_ANNO = True
    image_dir = '/home/jario/car-landing-data/200m/scaled_images'
    roi_saving_dir = '/home/jario/car-landing-data/our_car'
    object_cnt = 28000

    args = parser.parse_args()
    image_jsons = open_spire_annotations(args.spire_anno)
    print(args.classes)

    if os.path.exists(os.path.join(args.spire_anno, 'annotations')):
        args.spire_anno = os.path.join(args.spire_anno, 'annotations')

    for image_anno in image_jsons:
        retained_annos = []
        for anno in image_anno['annos']:
            bbox = np.array(anno['bbox'], np.float32)
            retain = True
            if bbox[2] == 0 or bbox[3] == 0:
                print("File_name: {}, bbox: {}".format(image_anno['file_name'], bbox))
                retain = False
            # assert 'score' in anno.keys(), "Annotations should have score."
            if anno['category_name'] not in args.classes:
                print("File_name: {}, cat: {}".format(image_anno['file_name'], anno['category_name']))
                retain = False
            if retain:
                retained_annos.append(anno)
                
                # 是否保存该类别的RoI区域，进行后续分类等任务
                if SAVING_ROI:
                    object_cnt += 1
                    image = cv2.imread(os.path.join(image_dir, image_anno['file_name']))
                    h, w, _ = image.shape
                    x1, y1, x2, y2 = int(anno['bbox'][0]), int(anno['bbox'][1]), \
                                     int(anno['bbox'][0] + anno['bbox'][2] - 1), \
                                     int(anno['bbox'][1] + anno['bbox'][3] - 1)
                    bw = anno['bbox'][2]
                    bh = anno['bbox'][3]
                    cx = x1 + bw / 2.
                    cy = y1 + bh / 2.
                    # bwm = max(bh, bw) * 0.5
                    # bhm = max(bh, bw) * 0.5
                    bwm = bw * 0.572
                    bhm = bh * 0.572
                    x1, y1, x2, y2 = int(round(cx - bwm)), int(round(cy - bhm)), \
                                     int(round(cx + bwm)), int(round(cy + bhm))
                    x1 = max(x1, 0)
                    y1 = max(y1, 0)
                    x2 = min(x2, w - 1)
                    y2 = min(y2, h - 1)

                    roi_image = image[y1:y2, x1:x2]
                    roi_image = cv2.resize(roi_image, (256, 256))
                    cv2.imshow('roi_image', roi_image)
                    cv2.waitKey(200)

                    cls_label = np.array([1, 1], np.int64)
                    # cls_label = np.array([0, 0], np.int64)

                    image_path = os.path.join(roi_saving_dir, 'images')  # images
                    if not os.path.exists(image_path):
                        os.mkdir(image_path)
                    cv2.imwrite(os.path.join(image_path, str(object_cnt).zfill(8) + '.jpg'), roi_image)

                    label_path = os.path.join(roi_saving_dir, 'labels')
                    if not os.path.exists(label_path):
                        os.mkdir(label_path)
                    np.save(os.path.join(label_path, str(object_cnt).zfill(8)), cls_label)

        image_anno['annos'] = retained_annos
        
        if SAVING_SPIRE_ANNO:
            fp = open(os.path.join(args.spire_anno, image_anno['file_name']+'.json'), 'w')
            json.dump(image_anno, fp)


if __name__ == '__main__':
    main()
