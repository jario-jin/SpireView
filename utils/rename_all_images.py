
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
        default="F:/BB220406-Cities-Skylines-train-1200-split/annotations",
        help="path to spire annotation dir",
        # required=True
    )
    parser.add_argument(
        "--spire-img",
        default="F:/BB220406-Cities-Skylines-train-1200-split/scaled_images",
        help="path to spire images dir",
        # required=True
    )
    parser.add_argument(
        "--output-dir",
        default="F:/BB220406-Cities-Skylines-val-1200",
        help="path to output dir",
        # required=True
    )
    args = parser.parse_args()

    output_img_fn = os.path.join(args.output_dir, "scaled_images")
    output_ann_fn = os.path.join(args.output_dir, "annotations")
    if not os.path.exists(output_img_fn):
        os.makedirs(output_img_fn)
        os.makedirs(output_ann_fn)

    for dir_nm in os.listdir(args.spire_anno):
        print(dir_nm)
        dir_nm_fn = os.path.join(args.spire_anno, dir_nm)

        cnt = 1
        for f in os.listdir(dir_nm_fn):
            if f.endswith('.json'):
                json_fn = os.path.join(args.spire_anno, dir_nm, f)
                img_fn = os.path.join(args.spire_img, dir_nm, f[:-5])
                print(json_fn)
                print(img_fn)
                rename = "{}-220404-{}.jpg".format(dir_nm, str(cnt).zfill(6))
                print(rename)

                json_f = open(json_fn, 'r')
                json_str = json_f.read()
                json_dict = json.loads(json_str)
                json_dict['file_name'] = rename

                img_attrs = {"side-view": 0, "bird-view": 0,
                             "high-alt": 0, "medium-alt": 0, "low-alt": 0,
                             "daylight": 0, "night": 0}
                if 'HA-BV-L' == dir_nm:
                    img_attrs["high-alt"] = 1
                    img_attrs["bird-view"] = 1
                    img_attrs["daylight"] = 1
                if 'HA-BV-N' == dir_nm:
                    img_attrs["high-alt"] = 1
                    img_attrs["bird-view"] = 1
                    img_attrs["night"] = 1
                if 'HA-SV-L' == dir_nm:
                    img_attrs["high-alt"] = 1
                    img_attrs["side-view"] = 1
                    img_attrs["daylight"] = 1
                if 'HA-SV-N' == dir_nm:
                    img_attrs["high-alt"] = 1
                    img_attrs["side-view"] = 1
                    img_attrs["night"] = 1

                if 'LA-BV-L' == dir_nm:
                    img_attrs["low-alt"] = 1
                    img_attrs["bird-view"] = 1
                    img_attrs["daylight"] = 1
                if 'LA-BV-N' == dir_nm:
                    img_attrs["low-alt"] = 1
                    img_attrs["bird-view"] = 1
                    img_attrs["night"] = 1
                if 'LA-SV-L' == dir_nm:
                    img_attrs["low-alt"] = 1
                    img_attrs["side-view"] = 1
                    img_attrs["daylight"] = 1
                if 'LA-SV-N' == dir_nm:
                    img_attrs["low-alt"] = 1
                    img_attrs["side-view"] = 1
                    img_attrs["night"] = 1

                if 'MA-BV-L' == dir_nm:
                    img_attrs["medium-alt"] = 1
                    img_attrs["bird-view"] = 1
                    img_attrs["daylight"] = 1
                if 'MA-BV-N' == dir_nm:
                    img_attrs["medium-alt"] = 1
                    img_attrs["bird-view"] = 1
                    img_attrs["night"] = 1
                if 'MA-SV-L' == dir_nm:
                    img_attrs["medium-alt"] = 1
                    img_attrs["side-view"] = 1
                    img_attrs["daylight"] = 1
                if 'MA-SV-N' == dir_nm:
                    img_attrs["medium-alt"] = 1
                    img_attrs["side-view"] = 1
                    img_attrs["night"] = 1

                json_dict['img_attrs'] = img_attrs
                with open(os.path.join(output_ann_fn, rename + '.json'), 'w') as fff:
                    json.dump(json_dict, fff)

                open(os.path.join(output_img_fn, rename), 'wb').write(
                    open(img_fn, 'rb').read())

                cnt += 1
                print('done!')


if __name__ == '__main__':
    main()
