
import argparse
import os
import json
import cv2
import numpy as np
from annotation_stat import open_spire_annotations
from shutil import copyfile


def main():
    parser = argparse.ArgumentParser(description="Statistic on spire annotations")
    parser.add_argument(
        "--spire-anno",
        default="C:/dataset/spire_dataset/BB211205-visdrone19-val",
        help="path to spire annotation dir",
        # required=True
    )
    args = parser.parse_args()
    image_jsons = open_spire_annotations(args.spire_anno)

    scaled_images_dir = os.path.join(args.spire_anno, 'scaled_images')
    annotations_dir = os.path.join(args.spire_anno, 'annotations')

    for image_anno in image_jsons:
        high_alt = image_anno['img_attrs']['high-alt']
        medium_alt = 0  # image_anno['img_attrs']['medium-alt']
        low_alt = image_anno['img_attrs']['low-alt']
        bird_view = image_anno['img_attrs']['bird-view']
        front_view = image_anno['img_attrs']['front-view']
        side_view = image_anno['img_attrs']['side-view']
        daylight = image_anno['img_attrs']['daylight']
        night = image_anno['img_attrs']['night']
        fog = image_anno['img_attrs']['fog']

        file_name = image_anno['file_name']
        scaled_images_fn = os.path.join(scaled_images_dir, file_name)
        annotations_fn = os.path.join(annotations_dir, file_name + '.json')

        if high_alt:
            scaled_images_fn_high_alt = os.path.join(args.spire_anno, 'attrs_high_alt', 'scaled_images')
            annotations_fn_high_alt = os.path.join(args.spire_anno, 'attrs_high_alt', 'annotations')
            if not os.path.exists(scaled_images_fn_high_alt):
                os.makedirs(scaled_images_fn_high_alt)
                os.makedirs(annotations_fn_high_alt)
            copyfile(scaled_images_fn, os.path.join(scaled_images_fn_high_alt, file_name))
            copyfile(annotations_fn, os.path.join(annotations_fn_high_alt, file_name + '.json'))

        if medium_alt:
            scaled_images_fn_medium_alt = os.path.join(args.spire_anno, 'attrs_medium_alt', 'scaled_images')
            annotations_fn_medium_alt = os.path.join(args.spire_anno, 'attrs_medium_alt', 'annotations')
            if not os.path.exists(scaled_images_fn_medium_alt):
                os.makedirs(scaled_images_fn_medium_alt)
                os.makedirs(annotations_fn_medium_alt)
            copyfile(scaled_images_fn, os.path.join(scaled_images_fn_medium_alt, file_name))
            copyfile(annotations_fn, os.path.join(annotations_fn_medium_alt, file_name + '.json'))

        if low_alt:
            scaled_images_fn_low_alt = os.path.join(args.spire_anno, 'attrs_low_alt', 'scaled_images')
            annotations_fn_low_alt = os.path.join(args.spire_anno, 'attrs_low_alt', 'annotations')
            if not os.path.exists(scaled_images_fn_low_alt):
                os.makedirs(scaled_images_fn_low_alt)
                os.makedirs(annotations_fn_low_alt)
            copyfile(scaled_images_fn, os.path.join(scaled_images_fn_low_alt, file_name))
            copyfile(annotations_fn, os.path.join(annotations_fn_low_alt, file_name + '.json'))

        if bird_view:
            scaled_images_fn_bird_view = os.path.join(args.spire_anno, 'attrs_bird_view', 'scaled_images')
            annotations_fn_bird_view = os.path.join(args.spire_anno, 'attrs_bird_view', 'annotations')
            if not os.path.exists(scaled_images_fn_bird_view):
                os.makedirs(scaled_images_fn_bird_view)
                os.makedirs(annotations_fn_bird_view)
            copyfile(scaled_images_fn, os.path.join(scaled_images_fn_bird_view, file_name))
            copyfile(annotations_fn, os.path.join(annotations_fn_bird_view, file_name + '.json'))

        if front_view:
            scaled_images_fn_front_view = os.path.join(args.spire_anno, 'attrs_front_view', 'scaled_images')
            annotations_fn_front_view = os.path.join(args.spire_anno, 'attrs_front_view', 'annotations')
            if not os.path.exists(scaled_images_fn_front_view):
                os.makedirs(scaled_images_fn_front_view)
                os.makedirs(annotations_fn_front_view)
            copyfile(scaled_images_fn, os.path.join(scaled_images_fn_front_view, file_name))
            copyfile(annotations_fn, os.path.join(annotations_fn_front_view, file_name + '.json'))

        if side_view:
            scaled_images_fn_side_view = os.path.join(args.spire_anno, 'attrs_side_view', 'scaled_images')
            annotations_fn_side_view = os.path.join(args.spire_anno, 'attrs_side_view', 'annotations')
            if not os.path.exists(scaled_images_fn_side_view):
                os.makedirs(scaled_images_fn_side_view)
                os.makedirs(annotations_fn_side_view)
            copyfile(scaled_images_fn, os.path.join(scaled_images_fn_side_view, file_name))
            copyfile(annotations_fn, os.path.join(annotations_fn_side_view, file_name + '.json'))

        if daylight:
            scaled_images_fn_daylight = os.path.join(args.spire_anno, 'attrs_daylight', 'scaled_images')
            annotations_fn_daylight = os.path.join(args.spire_anno, 'attrs_daylight', 'annotations')
            if not os.path.exists(scaled_images_fn_daylight):
                os.makedirs(scaled_images_fn_daylight)
                os.makedirs(annotations_fn_daylight)
            copyfile(scaled_images_fn, os.path.join(scaled_images_fn_daylight, file_name))
            copyfile(annotations_fn, os.path.join(annotations_fn_daylight, file_name + '.json'))

        if night:
            scaled_images_fn_night = os.path.join(args.spire_anno, 'attrs_night', 'scaled_images')
            annotations_fn_night = os.path.join(args.spire_anno, 'attrs_night', 'annotations')
            if not os.path.exists(scaled_images_fn_night):
                os.makedirs(scaled_images_fn_night)
                os.makedirs(annotations_fn_night)
            copyfile(scaled_images_fn, os.path.join(scaled_images_fn_night, file_name))
            copyfile(annotations_fn, os.path.join(annotations_fn_night, file_name + '.json'))

        if fog:
            scaled_images_fn_fog = os.path.join(args.spire_anno, 'attrs_fog', 'scaled_images')
            annotations_fn_fog = os.path.join(args.spire_anno, 'attrs_fog', 'annotations')
            if not os.path.exists(scaled_images_fn_fog):
                os.makedirs(scaled_images_fn_fog)
                os.makedirs(annotations_fn_fog)
            copyfile(scaled_images_fn, os.path.join(scaled_images_fn_fog, file_name))
            copyfile(annotations_fn, os.path.join(annotations_fn_fog, file_name + '.json'))
        
        # fp = open(os.path.join(args.spire_anno, image_anno['file_name']+'.json'), 'w')
        # json.dump(image_anno, fp)


if __name__ == '__main__':
    main()
