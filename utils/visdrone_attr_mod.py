
import argparse
import os
import json
import cv2
import numpy as np
from annotation_stat import open_spire_annotations


SAVING_SPIRE_ANNO = True


def main():
    parser = argparse.ArgumentParser(description="Statistic on spire annotations")
    parser.add_argument(
        "--spire-anno",
        default="C:/dataset/spire_dataset/BB210520-visdrone19-train/annotations",
        help="path to spire annotation dir",
        # required=True
    )

    args = parser.parse_args()
    image_jsons = open_spire_annotations(args.spire_anno)

    if os.path.exists(os.path.join(args.spire_anno, 'annotations')):
        args.spire_anno = os.path.join(args.spire_anno, 'annotations')

    for image_anno in image_jsons:
        img_attrs = image_anno['img_attrs']
        if img_attrs['low-alt'] == 1 or img_attrs['medium-alt'] == 1:
            img_attrs['low-alt'] = 1

        img_attrs.pop('medium-alt')
        
        if SAVING_SPIRE_ANNO:
            fp = open(os.path.join(args.spire_anno, image_anno['file_name'] + '.json'), 'w')
            json.dump(image_anno, fp)


if __name__ == '__main__':
    main()
