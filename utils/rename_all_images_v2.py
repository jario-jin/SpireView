
import argparse
import os
import json
import cv2
import numpy as np
from annotation_stat import open_spire_annotations


def main():
    parser = argparse.ArgumentParser(description="Statistic on spire annotations")
    parser.add_argument(
        "--spire-dir",
        default="/media/jario/JarioT7/dataset/spire_dataset/BB-Cities-Skylines-val-v220509",
        help="path to spire annotation dir",
        # required=True
    )
    parser.add_argument(
        "--output-dir",
        default="F:/BB220406-Cities-Skylines-val-1200",
        help="path to output dir",
        # required=True
    )
    args = parser.parse_args()

    # output_img_fn = os.path.join(args.output_dir, "scaled_images")
    # output_ann_fn = os.path.join(args.output_dir, "annotations")
    # if not os.path.exists(output_img_fn):
    #     os.makedirs(output_img_fn)
    #     os.makedirs(output_ann_fn)

    for dir_nm in os.listdir(args.spire_dir):
        print(dir_nm)
        sub_dir = os.path.join(args.spire_dir, dir_nm)
        if not os.path.isdir(sub_dir):
            continue
        dir_nm_fn = os.path.join(args.spire_dir, dir_nm, "annotations")

        cnt = 1
        for f in os.listdir(dir_nm_fn):
            if f.endswith('.json'):
                json_fn = os.path.join(dir_nm_fn, f)
                img_fn = os.path.join(args.spire_dir, dir_nm, "scaled_images", f[:-5])
                print(json_fn)
                print(img_fn)
                # rename = "{}-220404-{}.jpg".format(dir_nm, str(cnt).zfill(6))
                # print(rename)

                json_f = open(json_fn, 'r')
                json_str = json_f.read()
                json_dict = json.loads(json_str)
                # json_dict['file_name'] = rename

                img_attrs = {"side-view": 0, "bird-view": 0,
                             "high-alt": 0, "medium-alt": 0, "low-alt": 0,
                             "daylight": 0, "night": 0}
                if 'HA_BV_L' == dir_nm:
                    img_attrs["high-alt"] = 1
                    img_attrs["bird-view"] = 1
                    img_attrs["daylight"] = 1
                if 'HA_BV_N' == dir_nm:
                    img_attrs["high-alt"] = 1
                    img_attrs["bird-view"] = 1
                    img_attrs["night"] = 1
                if 'HA_SV_L' == dir_nm:
                    img_attrs["high-alt"] = 1
                    img_attrs["side-view"] = 1
                    img_attrs["daylight"] = 1
                if 'HA_SV_N' == dir_nm:
                    img_attrs["high-alt"] = 1
                    img_attrs["side-view"] = 1
                    img_attrs["night"] = 1

                if 'LA_BV_L' == dir_nm:
                    img_attrs["low-alt"] = 1
                    img_attrs["bird-view"] = 1
                    img_attrs["daylight"] = 1
                if 'LA_BV_N' == dir_nm:
                    img_attrs["low-alt"] = 1
                    img_attrs["bird-view"] = 1
                    img_attrs["night"] = 1
                if 'LA_SV_L' == dir_nm:
                    img_attrs["low-alt"] = 1
                    img_attrs["side-view"] = 1
                    img_attrs["daylight"] = 1
                if 'LA_SV_N' == dir_nm:
                    img_attrs["low-alt"] = 1
                    img_attrs["side-view"] = 1
                    img_attrs["night"] = 1

                if 'MA_BV_L' == dir_nm:
                    img_attrs["medium-alt"] = 1
                    img_attrs["bird-view"] = 1
                    img_attrs["daylight"] = 1
                if 'MA_BV_N' == dir_nm:
                    img_attrs["medium-alt"] = 1
                    img_attrs["bird-view"] = 1
                    img_attrs["night"] = 1
                if 'MA_SV_L' == dir_nm:
                    img_attrs["medium-alt"] = 1
                    img_attrs["side-view"] = 1
                    img_attrs["daylight"] = 1
                if 'MA_SV_N' == dir_nm:
                    img_attrs["medium-alt"] = 1
                    img_attrs["side-view"] = 1
                    img_attrs["night"] = 1

                json_dict['img_attrs'] = img_attrs
                with open(json_fn, 'w') as fff:
                    json.dump(json_dict, fff)

                # open(os.path.join(output_img_fn, rename), 'wb').write(
                #     open(img_fn, 'rb').read())

                cnt += 1
                print('done!')


if __name__ == '__main__':
    main()
