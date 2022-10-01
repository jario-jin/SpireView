import argparse
import os
import json
import cv2
from tqdm import tqdm
import pycocotools.mask as maskUtils
import numpy as np
import cv2


def main():
    parser = argparse.ArgumentParser(description="Convert Taobao annotation to spire annotation")
    parser.add_argument(
        "--taobao-anno",
        default="C:/tmp/2020xian/ann",
        help="path to Taobao annotation dir",
    )
    parser.add_argument(
        "--taobao-image",
        default="C:/tmp/2020xian/img",
        help="path to Taobao image dir",
    )
    parser.add_argument(
        "--output-dir",
        default="D:/VID220315-BIT-Air2Air-train",
        help="path to spire home dir",
    )
    args = parser.parse_args()

    output_img_dir = os.path.join(args.output_dir, 'scaled_images')
    output_ann_dir = os.path.join(args.output_dir, 'annotations')
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_ann_dir):
        os.makedirs(output_ann_dir)

    vid_names = os.listdir(args.taobao_image)
    vid_names.sort()
    tracked_id = 0
    for i, vid_name in enumerate(vid_names):
        print("{}: {}".format(i, vid_name))
        img_names = os.listdir(os.path.join(args.taobao_image, vid_name))
        img_names.sort()
        tracked_id += 1
        for img_name in img_names:
            if img_name[-4:] == '.jpg':
                img = cv2.imread(os.path.join(args.taobao_image, vid_name, img_name))
                if img is None:
                    print('READ [{}] ERROR!'.format(os.path.join(args.taobao_image, vid_name, img_name)))
                    return -1

                h, w, c = img.shape
                if h > 1080:
                    scale = 1080 / h
                    h = 1080
                    w = int(round(w * scale))
                    img = cv2.resize(img, (w, h))
                else:
                    scale = 1.

                spire_dict = {}
                spire_dict['file_name'] = vid_name + '/' + img_name
                spire_dict['height'], spire_dict['width'] = h, w
                spire_dict['annos'] = []

                # reading json
                json_name = img_name[:-3] + "json"  # 注意这里img的后缀一定是.jpg
                if os.path.exists(os.path.join(args.taobao_anno, vid_name, json_name)):
                    json_f = open(os.path.join(args.taobao_anno, vid_name, json_name), 'r')
                    json_str = json_f.read()
                    json_dict = json.loads(json_str)
                    json_f.close()
                    assert len(json_dict['shapes']) == 1
                    for sp in json_dict['shapes']:
                        spire_anno = {}
                        spire_anno['tracked_id'] = tracked_id
                        x = sp['points'][0][0] * scale
                        y = sp['points'][0][1] * scale
                        w = sp['points'][1][0] * scale - sp['points'][0][0] * scale
                        h = sp['points'][1][1] * scale - sp['points'][0][1] * scale
                        spire_anno['area'] = w * h
                        spire_anno['bbox'] = [x, y, w, h]
                        spire_anno['segmentation'] = [[x, y, x, y+h, x+w, y+h, x+w, y]]
                        spire_anno['category_name'] = 'drone'
                        spire_dict['annos'].append(spire_anno)
                else:
                    print("  NON-EXIST: {}".format(json_name))

                if not os.path.exists(os.path.join(output_ann_dir, vid_name)):
                    os.makedirs(os.path.join(output_ann_dir, vid_name))
                if not os.path.exists(os.path.join(output_img_dir, vid_name)):
                    os.makedirs(os.path.join(output_img_dir, vid_name))

                with open(os.path.join(output_ann_dir, vid_name, img_name + '.json'), "w") as f:
                    json.dump(spire_dict, f)
                cv2.imwrite(os.path.join(output_img_dir, vid_name, img_name), img)


if __name__ == '__main__':
    main()
