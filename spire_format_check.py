import os
import cv2
import json


def load_spire_dir(spire_dir):
    """
    :param spire_dir: path to spire dataset root dir
    :return: List - img_list, List - ann_list
    """
    img_list = []
    ann_list = []
    for sub_dir in os.listdir(spire_dir):
        sub_fn = os.path.join(spire_dir, sub_dir)
        if os.path.isdir(sub_fn):
            if sub_dir == 'scaled_images':
                img_list.extend([os.path.join(sub_fn, img_nm) for img_nm in os.listdir(sub_fn) if
                                 img_nm.endswith('.jpg') or img_nm.endswith('.png')])
                ann_list.extend([os.path.join(spire_dir, 'annotations', img_nm + '.json') for img_nm in
                                 os.listdir(sub_fn) if
                                 img_nm.endswith('.jpg') or img_nm.endswith('.png')])
            else:
                for sub_sub_dir in os.listdir(sub_fn):
                    sub_sub_fn = os.path.join(sub_fn, sub_sub_dir)
                    if os.path.isdir(sub_sub_fn):
                        if sub_sub_dir == 'scaled_images':
                            img_list.extend([os.path.join(sub_sub_fn, img_nm) for img_nm in os.listdir(sub_sub_fn) if
                                             img_nm.endswith('.jpg') or img_nm.endswith('.png')])
                            ann_list.extend([os.path.join(sub_fn, 'annotations', img_nm + '.json') for img_nm in
                                             os.listdir(sub_sub_fn) if
                                             img_nm.endswith('.jpg') or img_nm.endswith('.png')])
    return img_list, ann_list


def format_check(spire_root_dir):
    img_list, ann_list = load_spire_dir(spire_root_dir)
    for img_fn, ann_fn in zip(img_list, ann_list):
        img = cv2.imread(img_fn)
        ann = json.loads(open(ann_fn, 'r').read())

    print('Check Done! Success!')


if __name__ == '__main__':
    spire_root = 'C:/Users/jario/Videos/BaiduYun/data'
    format_check(spire_root)

