#!/usr/bin/env python

import subprocess
import os
import glob
import argparse
import torch
from torchvision import transforms as T
import time
import logging
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from maskrcnn_benchmark.modeling.detector.detectors import build_detection_model
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

from spire_anno import SpireAnno
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms


# load config from file and command-line arguments
config_file = '/home/jario/spire-net-1905/maskrcnn-benchmark-models/pre_trained/coco_frcnn_r50_fpn_ms.yaml'
weight_file = '/home/jario/spire-net-1905/maskrcnn-benchmark-models/pre_trained/coco_frcnn_r50_fpn_ms.pth'
templete = 'coco'
output_dir = '/tmp/coco'
image_dir = '/home/jario/dataset/coco/minival2014'
gt = '/home/jario/dataset/coco/annotations/instances_minival2014.json'
min_image_size = (800,)

feats_saving_dir = '/tmp/fcos_f_cocominival'


def build_transform(cfg, min_image_size):
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(min_image_size),
            T.ToTensor(),
            T.Lambda(lambda x: x * 255),
            normalize_transform,
        ]
    )
    return transform


def build_transform_origin(cfg):
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.ToTensor(),
            T.Lambda(lambda x: x * 255),
            normalize_transform,
        ]
    )
    return transform


class SpireDetector(object):
    """
    spire目标检测器
    """
    def __init__(self, config_file, weight_file, min_image_size=(800,), dataset='coco', spire_dir='/tmp',
                 class_transfor=None, origin_size=False, logger=logging.getLogger()):
        # 加载模型配置文件
        cfg.merge_from_file(config_file)
        cfg.freeze()
        self.cfg = cfg
        model = build_detection_model(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)
        model.to(self.device)
        # 读取与加载网络参数
        checkpoint = torch.load(weight_file, map_location=torch.device("cpu"))
        load_state_dict(model, checkpoint.pop("model"))
        model.eval()
        self.model = model
        if origin_size:
            self.transforms = [build_transform_origin(cfg)]
        else:
            self.transforms = [build_transform(cfg, _size) for _size in min_image_size]
        self.spire_anno = SpireAnno(dataset=dataset, spire_dir=spire_dir)

    def _class_independ_nms(self, boxlist, nms_thresh):
        """
        类别独立的非极大抑制(nms)
        :param boxlist (BoxList): 输入检测目标框
        :param nms_thresh (float): nms阈值
        :return:
        """
        scores = boxlist.get_field("scores")
        labels = boxlist.get_field("labels")
        boxes = boxlist.bbox

        num_classes = self.spire_anno.num_classes
        result = []
        for i in range(1, num_classes + 1):  # boxlist含有背景类0
            inds = (labels == i).nonzero().view(-1)

            scores_i = scores[inds]
            boxes_i = boxes[inds, :].view(-1, 4)
            boxlist_for_class = BoxList(boxes_i, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_i)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, nms_thresh,
                score_field="scores"
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), i, dtype=torch.int64, device=scores.device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        return result

    def detect(self, image, saving_name='', verbose=False):
        """
        检测一幅opencv打开的图像
        :param image (np.ndarray): 输入图像
        :return: None
        """
        prediction_list = []
        for transform in self.transforms:
            predictions = self._inference_single_cvimage(image, transform, verbose=verbose)
            prediction_list.append(predictions)

        predictions = cat_boxlist(prediction_list)
        predictions = self._class_independ_nms(predictions, nms_thresh=0.6)

        if len(saving_name) > 0:
            self.spire_anno.from_maskrcnn_benchmark(predictions, f, image.shape)
        return predictions

    def _inference_single_cvimage(self, cv_image, transforms, verbose=False, score_th=0.2):
        nh, nw = cv_image.shape[:2]
        image = transforms(cv_image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)

        ## compute prediction
        with torch.no_grad():
            predictions = self.model(image_list)

        predictions = predictions[0].to("cpu")
        # reshape prediction (a BoxList) into the original image size
        predictions = predictions.resize((nw, nh))
        if verbose:
            vis_image = self.spire_anno.visualize_boxlist(cv_image, predictions, score_th)
            cv2.imshow("detected_image", vis_image)
            cv2.waitKey(10)

        return predictions


if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = logging.getLogger()
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(os.path.join(output_dir, "log_infer.txt"))
    fmt = '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s '
    stream_handler.setFormatter(logging.Formatter(fmt))
    stream_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt))
    file_handler.setLevel(logging.DEBUG)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    detector = SpireDetector(config_file, weight_file, min_image_size=min_image_size,
                             spire_dir=output_dir, dataset=templete)

    for f in os.listdir(image_dir):
        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.JPG'):
            image_fn = os.path.join(image_dir, f)
            image = cv2.imread(image_fn)
            start_time = time.time()

            detector.detect(image, f, verbose=True)

            print("Time: {:.2f} s / img".format(time.time() - start_time))
            if cv2.waitKey(1) == 27:
                break  # esc to quit

    cv2.destroyAllWindows()
    if gt is not None:
        eval_res = detector.spire_anno.cocoapi_eval(gt)

    print("all done...")
