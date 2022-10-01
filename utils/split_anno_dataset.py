
import argparse
import os
import json
import cv2
import random
import numpy as np
from annotation_stat import open_spire_annotations
from shutil import copyfile


def main():
    parser = argparse.ArgumentParser(description="Statistic on spire annotations")
    parser.add_argument(
        "--spire-dir",
        default="/home/jario/dataset/OUTPUT/target_domain_train",
        help="path to spire dir",
        # required=True
    )
    parser.add_argument(
        "--output-dir",
        default="/home/jario/dataset/OUTPUT",
        help="path to output dir",
        # required=True
    )
    parser.add_argument(
        "--none-dir",
        default="/home/jario/dataset/NONE",
        help="path to non-label samples output dir",
        # required=True
    )

    args = parser.parse_args()

    assert os.path.exists(os.path.join(args.spire_dir, 'annotations')), \
        "annotations not in {}".format(args.spire_dir)
    assert os.path.exists(os.path.join(args.spire_dir, 'scaled_images')), \
        "scaled_images not in {}".format(args.spire_dir)

    image_jsons = open_spire_annotations(
        os.path.join(args.spire_dir, 'annotations')
    )

    n_output_test = 1280

    allowed = []
    no_objs = []
    for i, image_anno in enumerate(image_jsons):

        if len(image_anno['annos']) == 0:
            print("[WARN] {} has no objects".format(image_anno['file_name']))
            no_objs.append(i)
        else:
            allowed.append(i)

    random.shuffle(allowed)
    test_ids = allowed[:n_output_test]
    train_ids = allowed[n_output_test:]

    test_annotations_dir = os.path.join(args.output_dir, 'test', 'annotations')
    test_scaled_images_dir = os.path.join(args.output_dir, 'test', 'scaled_images')
    if not os.path.exists(test_annotations_dir):
        os.makedirs(test_annotations_dir)
    if not os.path.exists(test_scaled_images_dir):
        os.makedirs(test_scaled_images_dir)
    for i in test_ids:
        image_anno = image_jsons[i]
        file_name = image_anno['file_name']
        copyfile(
            os.path.join(args.spire_dir, 'annotations', file_name + '.json'),
            os.path.join(test_annotations_dir, file_name + '.json')
        )
        copyfile(
            os.path.join(args.spire_dir, 'scaled_images', file_name),
            os.path.join(test_scaled_images_dir, file_name)
        )

    train_annotations_dir = os.path.join(args.output_dir, 'train', 'annotations')
    train_scaled_images_dir = os.path.join(args.output_dir, 'train', 'scaled_images')
    if not os.path.exists(train_annotations_dir):
        os.makedirs(train_annotations_dir)
    if not os.path.exists(train_scaled_images_dir):
        os.makedirs(train_scaled_images_dir)
    for i in train_ids:
        image_anno = image_jsons[i]
        file_name = image_anno['file_name']
        copyfile(
            os.path.join(args.spire_dir, 'annotations', file_name + '.json'),
            os.path.join(train_annotations_dir, file_name + '.json')
        )
        copyfile(
            os.path.join(args.spire_dir, 'scaled_images', file_name),
            os.path.join(train_scaled_images_dir, file_name)
        )

    none_annotations_dir = os.path.join(args.none_dir, 'annotations')
    none_scaled_images_dir = os.path.join(args.none_dir, 'scaled_images')
    if not os.path.exists(none_annotations_dir):
        os.mkdir(none_annotations_dir)
    if not os.path.exists(none_scaled_images_dir):
        os.mkdir(none_scaled_images_dir)
    for i in no_objs:
        image_anno = image_jsons[i]
        file_name = image_anno['file_name']
        copyfile(
            os.path.join(args.spire_dir, 'annotations', file_name + '.json'),
            os.path.join(none_annotations_dir, file_name + '.json')
        )
        copyfile(
            os.path.join(args.spire_dir, 'scaled_images', file_name),
            os.path.join(none_scaled_images_dir, file_name)
        )


if __name__ == '__main__':
    main()
