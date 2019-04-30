
import argparse
import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO
import time


def main():
    parser = argparse.ArgumentParser(description="Statistic on spire annotations")
    parser.add_argument(
        "--coco-anno",
        default="",
        help="path to coco annotation dir",
        required=True
    )
    args = parser.parse_args()
    
    coco = COCO(args.coco_anno)
    ids = list(coco.imgs.keys())
    # sort indices for reproducible results
    ids = sorted(ids)
    json_category_id_to_contiguous_id = {
        v: i + 1 for i, v in enumerate(coco.getCatIds())
    }
    contiguous_category_id_to_json_id = {
        v: k for k, v in json_category_id_to_contiguous_id.items()
    }
    id_to_img_map = {k: v for k, v in enumerate(ids)}

    i = 0
    for key, val_d in coco.anns.items():
        if 'segmentation' in val_d.keys():
            val = val_d['segmentation']
            if isinstance(val, dict):
                print("i: {}, key: {}".format(i, key))
            else:
                if len(val) == 0 or len(val[0]) == 0:
                    raise Exception("[ZERO] key: {}, val: {}".format(key, val))
                else:
                    # print(coco.anns[key])
                    coco.annToMask(coco.anns[key])
                    print("i: {}, len(l1): {}, len(l2): {}".format(i, len(val), len(val[0])))
        else:
            raise Exception("[NO_KEYS] key: {}, val: {}".format(key, val_d))
        i += 1
        # time.sleep(0.1)


if __name__ == '__main__':
    main()
