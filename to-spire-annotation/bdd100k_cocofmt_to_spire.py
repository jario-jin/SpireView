import argparse
import os
import json
import warnings
import cv2
from tqdm import tqdm
from collections import defaultdict
import pycocotools
from pycocotools.coco import COCO as _COCO
from pycocotools.cocoeval import COCOeval as _COCOeval
import pycocotools.mask as maskUtils
from pycocotools.coco import _isArrayLike
import numpy as np
import cv2


class COCO(_COCO):
    """This class is almost the same as official pycocotools package.

    It implements some snake case function aliases. So that the COCO class has
    the same interface as LVIS class.
    """

    def __init__(self, annotation_file=None):
        if getattr(pycocotools, '__version__', '0') >= '12.0.2':
            warnings.warn(
                'mmpycocotools is deprecated. Please install official pycocotools by "pip install pycocotools"',  # noqa: E501
                UserWarning)
        super().__init__(annotation_file=annotation_file)
        self.img_ann_map = self.imgToAnns
        self.cat_img_map = self.catToImgs

    def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
        return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)

    def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
        return self.getCatIds(cat_names, sup_names, cat_ids)

    def get_img_ids(self, img_ids=[], cat_ids=[]):
        return self.getImgIds(img_ids, cat_ids)

    def load_anns(self, ids):
        return self.loadAnns(ids)

    def load_cats(self, ids):
        return self.loadCats(ids)

    def load_imgs(self, ids):
        return self.loadImgs(ids)


class CocoVID(COCO):

    def __init__(self, annotation_file=None):
        assert annotation_file, 'Annotation file must be provided.'
        super(CocoVID, self).__init__(annotation_file=annotation_file)

    def createIndex(self):
        print('creating index...')
        anns, cats, imgs, vids = {}, {}, {}, {}
        imgToAnns, catToImgs, vidToImgs = defaultdict(list), defaultdict(
            list), defaultdict(list)

        if 'videos' in self.dataset:
            for video in self.dataset['videos']:
                vids[video['id']] = video

        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                vidToImgs[img['video_id']].append(img)
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
        self.videos = vids
        self.vidToImgs = vidToImgs

    def get_vid_ids(self, vidIds=[]):
        vidIds = vidIds if _isArrayLike(vidIds) else [vidIds]

        if len(vidIds) == 0:
            ids = self.videos.keys()
        else:
            ids = set(vidIds)

        return list(ids)

    def get_img_ids_from_vid(self, vidId):
        img_infos = self.vidToImgs[vidId]
        ids = list(np.zeros([len(img_infos)], dtype=np.int))
        for img_info in img_infos:
            ids[img_info['frame_id']] = img_info['id']
        return ids

    def load_vids(self, ids=[]):
        if _isArrayLike(ids):
            return [self.videos[id] for id in ids]
        elif type(ids) == int:
            return [self.videos[ids]]


def main():
    parser = argparse.ArgumentParser(description="Convert COCO annotation to spire annotation")
    parser.add_argument(
        "--coco-anno",
        default="C:/Users/jario/Downloads/converted_jsons/det_val_cocofmt.json",
        help="path to coco annotation file",
        # required=True
    )
    parser.add_argument(
        "--coco-image-dir",
        default="C:/tmp/bdd100k/images/100k/val",
        help="path to coco image dir",
    )
    parser.add_argument(
        "--output-dir",
        default="C:/dataset/SEG220309_bdd100k_det_val",
        help="path to spire home dir",
    )
    parser.add_argument(
        '--save-image',
        action='store_true',
        help='save image to scaled_images'
    )
    parser.add_argument(
        '--show-image',
        action='store_true',
        help='show image for testing'
    )
    args = parser.parse_args()

    image_dir = os.path.join(args.output_dir, 'scaled_images')
    anno_dir = os.path.join(args.output_dir, 'annotations')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)

    c_vid = CocoVID(annotation_file=args.coco_anno)
    c_vid.createIndex()

    # Pack all coco categories to a dictionary.
    category_id_to_name = {}
    for cat in c_vid.dataset['categories']:
        category_id_to_name[cat['id']] = cat['name']

    data_infos = []
    vid_ids = c_vid.get_vid_ids()
    img_ids = []
    for vid_id in vid_ids:
        img_ids = c_vid.get_img_ids_from_vid(vid_id)
        # interval = 1
        # img_ids = img_ids[::interval]
        assert c_vid.videos[vid_id]['id'] == vid_id
        vid_name = c_vid.videos[vid_id]['name']

        vid_image_dir = os.path.join(image_dir, vid_name)
        vid_anno_dir = os.path.join(anno_dir, vid_name)
        if not os.path.exists(vid_image_dir):
            os.makedirs(vid_image_dir)
        if not os.path.exists(vid_anno_dir):
            os.makedirs(vid_anno_dir)

        for img_id in img_ids:
            info = c_vid.load_imgs([img_id])[0]
            if len(info['file_name'].split('/')) > 2:
                replace_token = info['file_name'].split('/')[0] + '/' + info['file_name'].split('/')[1] + '/'
                info['file_name'] = info['file_name'].replace(replace_token, info['file_name'].split('/')[0] + '/')
            info['filename'] = info['file_name']
            data_infos.append(info)

            spire_dict = {}
            spire_dict['file_name'] = info['file_name']
            spire_dict['height'], spire_dict['width'] = info['height'], info['width']
            spire_dict['video_id'] = info['video_id']
            spire_dict['frame_id'] = info['frame_id']
            spire_dict['id'] = info['id']
            spire_dict['annos'] = []

            ann_ids = c_vid.get_ann_ids(img_id)
            if info['file_name'] == 'b1d22ed6-f1cac061/b1d22ed6-f1cac061-0000198.jpg':
                print('debug')

            for ann_id in ann_ids:
                ann_info = c_vid.load_anns([ann_id])[0]

                spire_anno = {}
                spire_anno['area'] = ann_info['area']
                spire_anno['bbox'] = ann_info['bbox']
                spire_anno['id'] = ann_info['id']
                spire_anno['image_id'] = ann_info['image_id']
                spire_anno['instance_id'] = ann_info['instance_id']
                spire_anno['category_id'] = ann_info['category_id']
                spire_anno['scalabel_id'] = ann_info['scalabel_id']
                spire_anno['iscrowd'] = ann_info['iscrowd']
                spire_anno['ignore'] = ann_info['ignore']
                spire_anno['segmentation'] = ann_info['segmentation']
                spire_anno['category_name'] = category_id_to_name[ann_info['category_id']]

                spire_dict['annos'].append(spire_anno)
                # print('debug')

            # Generate spire annotation files for each image
            output_fn = os.path.join(anno_dir, info['file_name'] + '.json')
            with open(output_fn, "w") as f:
                json.dump(spire_dict, f)

            open(os.path.join(image_dir, info['file_name']), 'wb').write(
                open(os.path.join(args.coco_image_dir, info['file_name']), 'rb').read())

    print('debug')


if __name__ == '__main__':
    main()
