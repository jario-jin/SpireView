
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
        default="/tmp/coco_spire",
        help="path to spire annotation dir",
        required=True
    )
    args = parser.parse_args()
    image_jsons = open_spire_annotations(args.spire_anno)

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
            segs = np.array(anno['segmentation'], np.float32)
            if len(segs) == 0:
                print("File_name: {}, segs: {}".format(image_anno['file_name'], segs))
                retain = False
            if retain:
                retained_annos.append(anno)
        image_anno['annos'] = retained_annos
        
        fp = file(os.path.join(args.spire_anno, image_anno['file_name']+'.json'), 'w')
        json.dump(image_anno, fp)

if __name__ == '__main__':
    main()
