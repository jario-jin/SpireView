# Copyright (c) Spire, Inc. All Rights Reserved.
# generate ap map, see 'Dynamic Zoom-in Network for Fast Object Detection in Large Images'
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

from spire_anno import find_contours, SpireAnno
from maskrcnn_detector import SpireDetector


# 判断目标是否被真实框召回的阈值
iou_thrs = 0.5
iou_miss_thrs = 0.3
# 将agmap放大的倍数(alpha)
alpha = 200.
# 生成agmap的大小
agmap_size = (64, 64)
# 输入特征的大小
c5_size = (16, 16)
# agmap均值和方差
agmap_mean = [0.003275, 0.002311]
agmap_std = [0.028399, 0.008956]


def open_spire_annotations(spire_dir):
    if os.path.exists(os.path.join(spire_dir, 'annotations')):
        spire_dir = os.path.join(spire_dir, 'annotations')

    image_jsons = []
    fns = os.listdir(spire_dir)
    fns.sort()
    for f in fns:
        if f.endswith('.json'):
            json_f = open(os.path.join(spire_dir, f), 'r')
            json_str = json_f.read()
            json_dict = json.loads(json_str)
            image_jsons.append(json_dict)
    return image_jsons


def agmap_segmentation(agmap, image_size, min_size):
    """
    对预测的agmap进行分割，返回高分辨率图像的子窗口(sub-windows)
    :param agmap (np.ndarray): [32, 32]
    :param image_size (tuple): (width, hegiht),图像的真实分辨率
    :param min_size (int): 输入网络的图像最小边长
    :return: 子窗口 [(xywh),...]
    """
    w, h = image_size
    if w > h:
        nh, nw = min_size, int(min_size * (float(w) / h))
    else:
        nw, nh = min_size, int(min_size * (float(h) / w))

    agmap = (agmap + 1) * 127
    agmap[agmap > 255] = 255
    agmap[agmap < 0] = 0
    agmap = agmap.astype(np.uint8)

    ## [bin]
    # Create binary image from source image
    _, bw = cv2.threshold(agmap, 145, 255, cv2.THRESH_BINARY)
    bw = cv2.resize(bw, (nw, nh))
    ## [bin]

    # Find total markers
    contours, _ = find_contours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        bbox = cv2.boundingRect(contour)
        # cv2.rectangle(bw, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255,255,255))
        bboxes.append(bbox)

    if 0 and len(bboxes) > 0:
        bboxes = np.asarray(bboxes, dtype=np.float32)
        boxlist = BoxList(bboxes, (nw, nh), mode='xywh')

    # Show the image
    # cv2.imshow('Binary Image', bw)
    return bboxes, (nw, nh)


agmap_avgpool = nn.AdaptiveAvgPool2d(agmap_size)


def agmap_coarse(gt_boxlist, l_boxlist, class_independ=False, keep_small=True, verbose=False):
    """
    利用真实值和低分辨率检测结果，生成agmap
    :param gt_boxlist (BoxList): 真实目标框，必须是xyxy类型
    :param l_boxlist (BoxList): 低分辨率检测结果，必须是xyxy类型
    :param class_independ (bool): 是否类别无关，只考虑proposal之间的iou
    :param keep_small (bool): 只计算小目标(< 96x96)的agmap
    :return: agmap (np.ndarray)
    """
    # 是否去除大目标，只计算96x96以下目标的agmap
    if keep_small:
        gt_area = gt_boxlist.area()
        l_area = l_boxlist.area()
        gt_keep, l_keep = gt_area < np.square(96), l_area < np.square(96)
        if torch.sum(gt_keep) == 0:
            gt_boxlist = BoxList([[0, 0, 0, 0]], gt_boxlist.size, mode="xyxy")
            gt_boxlist.add_field("labels", torch.as_tensor([0], dtype=torch.int64))
        else:
            gt_boxlist = gt_boxlist[gt_keep]
        if torch.sum(l_keep) == 0:
            l_boxlist = BoxList([[1, 1, 1, 1]], l_boxlist.size, mode="xyxy")
            l_boxlist.add_field("labels", torch.as_tensor([0], dtype=torch.int64))
            l_boxlist.add_field("scores", torch.as_tensor([0], dtype=torch.float32))
        else:
            l_boxlist = l_boxlist[l_keep]
    # 初始化agmap
    gt_w, gt_h = gt_boxlist.size
    agmap = np.zeros((2, gt_h, gt_w), np.float32)

    for i in range(len(gt_boxlist)):
        g_bbox_i = gt_boxlist[i]
        g_label = g_bbox_i.get_field("labels").item()

        if class_independ:  # 是否类别无关
            l_boxlist_sel = l_boxlist
        else:
            l_boxlist_sel = l_boxlist[l_boxlist.get_field("labels") == g_label]  # 正确召回的类别
            if len(l_boxlist_sel) == 0:
                l_boxlist_sel = BoxList([[1, 1, 1, 1]], l_boxlist_sel.size, mode="xyxy")
                l_boxlist_sel.add_field("scores", torch.as_tensor([0], dtype=torch.float32))

        l_score = l_boxlist_sel.get_field("scores").cpu().numpy()

        iou_l = boxlist_iou(g_bbox_i, l_boxlist_sel)
        l_val, l_id = iou_l.max(dim=1)
        l_val, l_id = l_val.item(), l_id.item()  # g_bbox_i只有一个元素

        g_bbox = g_bbox_i.bbox[0, :].cpu().numpy()
        g_bbox = np.round(g_bbox).astype(np.int64)  # 取整，以便索引
        g_area = (g_bbox[3] - g_bbox[1]) * (g_bbox[2] - g_bbox[0])

        l_bbox = l_boxlist_sel.bbox[l_id, :].cpu().numpy()
        l_bbox = np.round(l_bbox).astype(np.int64)
        l_area = (l_bbox[3] - l_bbox[1]) * (l_bbox[2] - l_bbox[0])

        if l_val > iou_thrs and g_area != 0:
            agmap[0, g_bbox[1]:g_bbox[3], g_bbox[0]:g_bbox[2]] += (1 - l_score[l_id]) / g_area
        elif g_area != 0:
            agmap[0, g_bbox[1]:g_bbox[3], g_bbox[0]:g_bbox[2]] += 1. / g_area

    iou_l = boxlist_iou(gt_boxlist, l_boxlist)

    l_score = l_boxlist.get_field("scores").cpu().numpy()
    l_label = l_boxlist.get_field("labels").cpu().numpy()
    g_label = gt_boxlist.get_field("labels").cpu().numpy()

    l_val, l_id = iou_l.max(dim=0)
    l_val, l_id = l_val.cpu().numpy(), l_id.cpu().numpy()

    for i in range(len(l_boxlist)):
        l_bbox = l_boxlist.bbox[i, :].cpu().numpy()
        l_bbox = np.round(l_bbox).astype(np.int64)  # 取整，以便索引
        area = (l_bbox[3] - l_bbox[1]) * (l_bbox[2] - l_bbox[0])
        if ((g_label[l_id[i]] != l_label[i] and not class_independ) or l_val[i] < iou_miss_thrs) and area != 0:
            agmap[1, l_bbox[1]:l_bbox[3], l_bbox[0]:l_bbox[2]] += l_score[i] / area  # 低分辨率误检收益

    agmap = torch.from_numpy(agmap).unsqueeze(dim=0)
    with torch.no_grad():
        # agmap = agmap_avgpool(agmap)
        agmap = F.interpolate(agmap, size=agmap_size, mode='bilinear', align_corners=False)
    agmap = np.squeeze(agmap.cpu().numpy())

    return agmap


def agmap_total(gt_boxlist, l_boxlist, h_boxlist, class_independ=False, keep_small=True,
                reward=False, verbose=False):
    """
    利用真实值，生成agmap
    :param gt_boxlist (BoxList): 真实目标框，必须是xyxy类型
    :param l_boxlist (BoxList): 低分辨率检测结果，必须是xyxy类型
    :param h_boxlist (BoxList): 高分辨率检测结果，必须是xyxy类型
    :param class_independ (bool): 是否类别无关，只考虑proposal之间的iou
    :param keep_small (bool): 只计算小目标(< 96x96)的agmap
    :return: agmap (np.ndarray)
    """
    # 是否去除大目标，只计算96x96以下目标的agmap
    if keep_small:
        gt_area = gt_boxlist.area()
        l_area = l_boxlist.area()
        h_area = h_boxlist.area()
        gt_keep, l_keep, h_keep = gt_area < np.square(96), l_area < np.square(96), h_area < np.square(96)
        if torch.sum(gt_keep) == 0:
            gt_boxlist = BoxList([[0, 0, 0, 0]], gt_boxlist.size, mode="xyxy")
            gt_boxlist.add_field("labels", torch.as_tensor([0], dtype=torch.int64))
        else:
            gt_boxlist = gt_boxlist[gt_keep]
        if torch.sum(l_keep) == 0:
            l_boxlist = BoxList([[0, 0, 0, 0]], l_boxlist.size, mode="xyxy")
            l_boxlist.add_field("labels", torch.as_tensor([0], dtype=torch.int64))
            l_boxlist.add_field("scores", torch.as_tensor([0], dtype=torch.float32))
        else:
            l_boxlist = l_boxlist[l_keep]
        if torch.sum(h_keep) == 0:
            h_boxlist = BoxList([[0, 0, 0, 0]], h_boxlist.size, mode="xyxy")
            h_boxlist.add_field("labels", torch.as_tensor([0], dtype=torch.int64))
            h_boxlist.add_field("scores", torch.as_tensor([0], dtype=torch.float32))
        else:
            h_boxlist = h_boxlist[h_keep]

    # gt_boxlist.size为(image_width, image_height)，转置后获得正确尺寸
    agmap = np.zeros(gt_boxlist.size, np.float32).T
    # 将收益分误检和漏检
    agmap_split = np.zeros((1, gt_boxlist.size[1], gt_boxlist.size[0]), np.float32)
    # 用于reward评价
    agval = 0.
    # 用ground-truth显示agmap，或者用l_det显示
    use_gt_bbox = True
    for i in range(len(gt_boxlist)):
        g_bbox_i = gt_boxlist[i]
        g_label = g_bbox_i.get_field("labels").item()

        if class_independ:
            l_boxlist_sel = l_boxlist
            h_boxlist_sel = h_boxlist
        else:
            l_boxlist_sel = l_boxlist[l_boxlist.get_field("labels") == g_label]  # 正确召回的类别
            h_boxlist_sel = h_boxlist[h_boxlist.get_field("labels") == g_label]
            if len(l_boxlist_sel) == 0:
                l_boxlist_sel = BoxList([[0, 0, 0, 0]], l_boxlist_sel.size, mode="xyxy")
                l_boxlist_sel.add_field("scores", torch.as_tensor([0], dtype=torch.float32))
            if len(h_boxlist_sel) == 0:
                h_boxlist_sel = BoxList([[0, 0, 0, 0]], h_boxlist_sel.size, mode="xyxy")
                h_boxlist_sel.add_field("scores", torch.as_tensor([0], dtype=torch.float32))

        l_score = l_boxlist_sel.get_field("scores").cpu().numpy()
        h_score = h_boxlist_sel.get_field("scores").cpu().numpy()

        iou_l = boxlist_iou(g_bbox_i, l_boxlist_sel)
        iou_h = boxlist_iou(g_bbox_i, h_boxlist_sel)

        l_val, l_id = iou_l.max(dim=1)
        l_val, l_id = l_val.item(), l_id.item()  # g_bbox_i只有一个元素
        h_val, h_id = iou_h.max(dim=1)
        h_val, h_id = h_val.item(), h_id.item()  # g_bbox_i只有一个元素

        # 首先根据ground-truth对agmap进行评分，分为3种情况，l和h都召回目标，l召回目标，h召回目标
        # g_bbox = gt_boxlist.bbox[i, :].cpu().numpy()
        g_bbox = g_bbox_i.bbox[0, :].cpu().numpy()
        g_bbox = np.round(g_bbox).astype(np.int64)  # 取整，以便索引
        g_area = (g_bbox[3] - g_bbox[1]) * (g_bbox[2] - g_bbox[0])

        l_bbox = l_boxlist_sel.bbox[l_id, :].cpu().numpy()
        l_bbox = np.round(l_bbox).astype(np.int64)
        l_area = (l_bbox[3] - l_bbox[1]) * (l_bbox[2] - l_bbox[0])

        if l_val > iou_thrs and h_val > iou_thrs:
            ag = h_score[h_id] - l_score[l_id]
        elif l_val > iou_thrs:  # 高分辨率漏检收益
            ag = -l_score[l_id]
        elif h_val > iou_thrs:  # 低分辨率漏检收益
            ag = h_score[h_id]
            if g_area != 0:
                agmap_split[0, g_bbox[1]:g_bbox[3], g_bbox[0]:g_bbox[2]] += ag / g_area
        else:
            ag = 0

        agval += ag
        if use_gt_bbox and g_area != 0:  # 使用ground-truth目标框来改变agmap的得分
            agmap[g_bbox[1]:g_bbox[3], g_bbox[0]:g_bbox[2]] += ag / g_area
        elif l_area != 0:
            agmap[l_bbox[1]:l_bbox[3], l_bbox[0]:l_bbox[2]] += ag / l_area

    iou_l = boxlist_iou(gt_boxlist, l_boxlist)
    iou_h = boxlist_iou(gt_boxlist, h_boxlist)

    l_score = l_boxlist.get_field("scores").cpu().numpy()
    h_score = h_boxlist.get_field("scores").cpu().numpy()
    l_label = l_boxlist.get_field("labels").cpu().numpy()
    h_label = h_boxlist.get_field("labels").cpu().numpy()

    g_label = gt_boxlist.get_field("labels").cpu().numpy()

    l_val, l_id = iou_l.max(dim=0)
    l_val, l_id = l_val.cpu().numpy(), l_id.cpu().numpy()
    h_val, h_id = iou_h.max(dim=0)
    h_val, h_id = h_val.cpu().numpy(), h_id.cpu().numpy()

    for i in range(len(l_boxlist)):
        l_bbox = l_boxlist.bbox[i, :].cpu().numpy()
        l_bbox = np.round(l_bbox).astype(np.int64)  # 取整，以便索引
        area = (l_bbox[3] - l_bbox[1]) * (l_bbox[2] - l_bbox[0])
        if ((g_label[l_id[i]] != l_label[i] and not class_independ) or l_val[i] < iou_thrs) and area != 0:
            agval += l_score[i]
            agmap[l_bbox[1]:l_bbox[3], l_bbox[0]:l_bbox[2]] += l_score[i] / area  # 低分辨率误检收益

    for i in range(len(h_boxlist)):
        h_bbox = h_boxlist.bbox[i, :].cpu().numpy()
        h_bbox = np.round(h_bbox).astype(np.int64)  # 取整，以便索引
        area = (h_bbox[3] - h_bbox[1]) * (h_bbox[2] - h_bbox[0])
        if ((g_label[h_id[i]] != h_label[i] and not class_independ) or h_val[i] < iou_thrs) and area != 0:
            agval -= h_score[i]
            agmap[h_bbox[1]:h_bbox[3], h_bbox[0]:h_bbox[2]] -= h_score[i] / area  # 高分辨率误检收益

    agmap = torch.from_numpy(agmap).unsqueeze(dim=0).unsqueeze(dim=0)
    agmap_split = torch.from_numpy(agmap_split).unsqueeze(dim=0)
    with torch.no_grad():
        # agmap = agmap_avgpool(agmap)
        agmap = F.interpolate(agmap, size=agmap_size, mode='bilinear', align_corners=False)
        agmap_split = F.interpolate(agmap_split, size=agmap_size, mode='bilinear', align_corners=False)
    agmap = np.squeeze(agmap.cpu().numpy())
    agmap_split = np.squeeze(agmap_split.cpu().numpy())

    if verbose:
        # 从[-1,1]转换到[0,255]，用以colormap可视化
        agmap_color = agmap * alpha
        agmap_color = cv2.resize(agmap_color, gt_boxlist.size)
        agmap_color = (agmap_color + 1) * 127
        agmap_color[agmap_color > 255] = 255
        agmap_color[agmap_color < 0] = 0
        agmap_color = cv2.applyColorMap(agmap_color.astype(np.uint8), cv2.COLORMAP_HOT)
        cv2.imshow("agmap", agmap_color)
        cv2.waitKey(200)

    if reward:
        return agval
    else:
        return agmap, agmap_split


def agmap_generate_test(agmap_saving_dir, spire_gt_dir, l_det_dir, dataset='coco', show_agmap=False):
    """
    读取spire标注信息生成apmap，测试样例
    :param agmap_saving_dir (str): agmap的保存路径
    :param show_agmap (bool): 是否显示agmap
    :return: None
    """
    spire_anno = SpireAnno(dataset=dataset)
    gt_list = open_spire_annotations(spire_gt_dir)

    for gt in gt_list:
        file_name = gt['file_name']
        print("file_name: {}".format(file_name))

        heigth, width = gt['height'], gt['width']
        l_det_fn = os.path.join(l_det_dir, file_name + '.json')
        # h_det_fn = os.path.join(h_det_dir, file_name + '.json')
        l_det = json.loads(open(l_det_fn, 'r').read())
        # h_det = json.loads(open(h_det_fn, 'r').read())

        assert heigth == l_det['height'] and width == l_det['width'], \
            "{}, height or width mismatch with ground-truth.".format(file_name)
        # assert heigth == h_det['height'] and width == h_det['width'], \
        #     "{}, height or width mismatch with ground-truth.".format(file_name)

        # 这里先认为类别无关
        gt_bbox = np.array([b['bbox'] for b in gt['annos']], np.float32)
        if len(gt_bbox) == 0:
            print("[WARNING]: no ground truth bounding-box.")
            agmap = np.zeros((2, agmap_size[0], agmap_size[1]), np.float32)
        else:
            # gt_boxlist = BoxList(gt_bbox, (width, heigth), mode="xywh").convert("xyxy")
            gt_boxlist = spire_anno.to_boxlist(gt)

            l_bbox = np.array([b['bbox'] for b in l_det['annos']], np.float32)
            if len(l_bbox) == 0:
                raise ValueError(
                    "no l_bbox detections, {}".format(file_name)
                )
            # l_score = np.array([b['score'] for b in l_det['annos']], np.float32)
            # l_boxlist = BoxList(l_bbox, (width, heigth), mode="xywh").convert("xyxy")
            # l_boxlist.add_field("scores", torch.as_tensor(l_score))
            l_boxlist = spire_anno.to_boxlist(l_det)

            '''
            h_bbox = np.array([b['bbox'] for b in h_det['annos']], np.float32)
            if len(h_bbox) == 0:
                raise ValueError(
                    "no h_bbox detections, {}".format(file_name)
                )
            # h_score = np.array([b['score'] for b in h_det['annos']], np.float32)
            # h_boxlist = BoxList(h_bbox, (width, heigth), mode="xywh").convert("xyxy")
            # h_boxlist.add_field("scores", torch.as_tensor(h_score))
            h_boxlist = spire_anno.to_boxlist(h_det)
            '''
            # agmap, agmap_sp = agmap_total(gt_boxlist, l_boxlist, h_boxlist)
            agmap = agmap_coarse(gt_boxlist, l_boxlist)

        np.save(os.path.join(agmap_saving_dir, file_name+'.npy'), agmap)
        if show_agmap:
            agmap_1 = agmap[0, :, :]
            agmap_1 = cv2.resize(agmap_1, (width, heigth))
            # 从[-1,1]转换到[0,255]，用以colormap可视化
            agmap_color = agmap_1 * alpha
            agmap_color = (agmap_color + 1) * 127
            agmap_color[agmap_color > 255] = 255
            agmap_color[agmap_color < 0] = 0
            agmap_color = cv2.applyColorMap(agmap_color.astype(np.uint8), cv2.COLORMAP_HOT)
            cv2.imshow("agmap_1", agmap_color)

            agmap_2 = agmap[1, :, :]
            agmap_2 = cv2.resize(agmap_2, (width, heigth))
            # 从[-1,1]转换到[0,255]，用以colormap可视化
            agmap_color = agmap_2 * alpha
            agmap_color = (agmap_color + 1) * 127
            agmap_color[agmap_color > 255] = 255
            agmap_color[agmap_color < 0] = 0
            agmap_color = cv2.applyColorMap(agmap_color.astype(np.uint8), cv2.COLORMAP_HOT)
            cv2.imshow("agmap_2", agmap_color)

            predicted_agmap_fn = os.path.join(predicted_agmap_dir, file_name + '.npy')
            if os.path.exists(predicted_agmap_fn):
                predicted_agmap = np.load(predicted_agmap_fn)
                predicted_agmap_1 = predicted_agmap[0, :, :]
                predicted_agmap_1 = cv2.resize(predicted_agmap_1, (width, heigth))
                agmap_color = (predicted_agmap_1 + 1) * 127
                agmap_color[agmap_color > 255] = 255
                agmap_color[agmap_color < 0] = 0

                agmap_color = cv2.applyColorMap(agmap_color.astype(np.uint8), cv2.COLORMAP_HOT)
                cv2.imshow("predicted_agmap_1", agmap_color)

                predicted_agmap_2 = predicted_agmap[1, :, :]
                predicted_agmap_2 = cv2.resize(predicted_agmap_2, (width, heigth))
                agmap_color = (predicted_agmap_2 + 1) * 127
                agmap_color[agmap_color > 255] = 255
                agmap_color[agmap_color < 0] = 0

                agmap_color = cv2.applyColorMap(agmap_color.astype(np.uint8), cv2.COLORMAP_HOT)
                cv2.imshow("predicted_agmap_2", agmap_color)

            cv2.waitKey(200)
            print('verbose')


from torch.utils.data import Dataset, DataLoader


class ResnetC5Dataset(Dataset):
    """载入Resnet中间层特征C5的numpy文件(x)，以及apmap的numpy文件(y)"""

    def __init__(self, resnet_c5_dir, agmap_dir):
        fns = os.listdir(agmap_dir)
        self.fns = fns
        self.resnet_c5_dir = resnet_c5_dir
        self.agmap_dir = agmap_dir
        # self.f_avgpool = nn.AdaptiveAvgPool2d((4, 4))

        data = np.load(os.path.join(self.resnet_c5_dir, self.fns[0]))
        self.channels = data.shape[1]

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        selected_fn = self.fns[idx]
        c5 = np.load(os.path.join(self.resnet_c5_dir, selected_fn))
        # 放大500倍
        agmap = np.load(os.path.join(self.agmap_dir, selected_fn)) * alpha

        n, c, h, w = c5.shape
        mapc, maph, mapw = agmap.shape
        # print("c5_ratio: {}, agmap_ratio: {}".format(w / h, mapw / maph))

        c5_torch = torch.from_numpy(c5)
        with torch.no_grad():
            # c5_torch = self.f_avgpool(c5_torch)
            c5_torch = F.interpolate(c5_torch, size=c5_size, mode='bilinear', align_corners=False)
            c5_torch = c5_torch.squeeze(dim=0)
        agmap_torch = torch.from_numpy(agmap)
        '''
        agmap_torch[0, :, :] -= agmap_mean[0]
        agmap_torch[1, :, :] -= agmap_mean[1]
        agmap_torch[0, :, :] /= agmap_std[0]
        agmap_torch[1, :, :] /= agmap_std[1]
        '''
        sample = {'c5': c5_torch, 'agmap': agmap_torch, 'file_name': selected_fn}
        return sample


class ResnetC5Collator(object):
    """
    将独立读取的c5和agmap组合起来，形成batch
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        h_max, w_max = 0, 0
        maph_max, mapw_max = 0, 0
        b = len(batch)
        file_names = []
        for sample in batch:
            c, h, w = sample['c5'].size()
            if h > h_max:
                h_max = h
            if w > w_max:
                w_max = w
            mc, h, w = sample['agmap'].size()
            if h > maph_max:
                maph_max = h
            if w > mapw_max:
                mapw_max = w
            file_names.append(sample['file_name'])

        c5_batch = torch.zeros((b, c, h_max, w_max), dtype=torch.float32)
        agmap_batch = torch.zeros((b, mc, maph_max, mapw_max), dtype=torch.float32)

        for i, sample in enumerate(batch):
            c, h, w = sample['c5'].size()
            c5_batch[i, :, :h, :w] = sample['c5']
            mc, h, w = sample['agmap'].size()
            agmap_batch[i, :, :h, :w] = sample['agmap']

        sample = {'c5': c5_batch, 'agmap': agmap_batch, 'file_name': file_names}
        return sample


class CRNet(nn.Module):
    def __init__(self, input_channels=2048):
        super(CRNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=2)
        self.gn1 = nn.GroupNorm(32, input_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.oh, self.ow = agmap_size[0], agmap_size[1]
        self.fc = nn.Linear(input_channels, self.oh * self.ow)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, 1, self.oh, self.ow)
        return x


class CRNetV2(nn.Module):
    def __init__(self, input_channels=2048):
        super(CRNetV2, self).__init__()
        middle_channels = 256
        layers = [
            nn.Conv2d(input_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        ]
        input_channels = middle_channels
        middle_channels = 128
        layers += [
            nn.ConvTranspose2d(input_channels, input_channels, kernel_size=2, stride=2),
            nn.Conv2d(input_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        ]
        input_channels = middle_channels
        middle_channels = 64
        layers += [
            nn.ConvTranspose2d(input_channels, input_channels, kernel_size=2, stride=2),
            nn.Conv2d(input_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        ]
        self.decode = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(middle_channels, 1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(middle_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.decode(x)
        a1 = self.conv1(x)
        a2 = self.conv2(x)
        a = torch.cat((a1, a2), dim=1)
        return a


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every default_downside_epoch epochs"""
    lr = base_lr * (0.1 ** (epoch // default_downside_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def cr_regressor_train(agmap_dir, c5_dir, cr_saving_fn):
    """
    训练能够生成apmap的网络，输入是resnet的c5特征
    :param agmap_dir (str): agmap的输入路径，文件夹
    :param c5_dir (str): c5特征的输入路径，文件夹
    :return:
    """
    train_epoch = 80
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ResnetC5Dataset(c5_dir, agmap_dir)
    collator = ResnetC5Collator()
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                            num_workers=4, collate_fn=collator)

    # model = CRNet(train_dataset.channels).to(device)
    model = CRNetV2(train_dataset.channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # weight_decay=1e-6
    # optimizer = optim.SGD(model.parameters(), base_lr, momentum=0.9, weight_decay=1e-6)

    model.train()
    for epoch in range(1, train_epoch + 1):
        for batch_idx, sample_batched in enumerate(dataloader):
            c5, agmap = sample_batched['c5'].to(device), sample_batched['agmap'].to(device)

            # adjust_learning_rate(optimizer, epoch)
            optimizer.zero_grad()
            # output = model(c5)
            output = model(c5)
            selected_pos = agmap != 0
            selected_neg = agmap == 0

            # print("1-{}".format(output.shape))
            # print("2-{}".format(selected_pos.shape))
            output_pos = output[selected_pos]
            output_neg = output[selected_neg]
            agmap_pos = agmap[selected_pos]
            agmap_neg = agmap[selected_neg]
            loss_pos = F.mse_loss(output_pos, agmap_pos, reduction='mean')
            loss_neg = F.mse_loss(output_neg, agmap_neg, reduction='mean')
            loss = loss_pos + loss_neg
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(c5), len(train_dataset),
                    100. * batch_idx / len(dataloader), loss.item()))

    # 保存模型
    torch.save({'state_dict': model.state_dict()}, cr_saving_fn)


def cr_regressor_inference(agmap_dir, c5_dir, cr_saving_fn, predicted_agmap_dir):
    """
    :param agmap_dir (str): agmap的输入路径，文件夹
    :param c5_dir (str): c5特征的输入路径，文件夹
    :return:
    """
    checkpoint = torch.load(cr_saving_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_dataset = ResnetC5Dataset(c5_dir, agmap_dir)
    dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                            num_workers=1)

    model = CRNetV2(eval_dataset.channels).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    mse_loss = 0
    avg_mean_1, avg_mean_2 = 0, 0
    avg_std_1, avg_std_2 = 0, 0
    for batch_idx, sample_batched in enumerate(dataloader):
        c5, agmap = sample_batched['c5'].to(device), sample_batched['agmap'].to(device)
        with torch.no_grad():
            output = model(c5)

        agmap_1 = agmap[0, 0, :, :].cpu().numpy()
        agmap_2 = agmap[0, 1, :, :].cpu().numpy()
        mean_1, mean_2 = np.mean(agmap_1), np.mean(agmap_2)
        std_1, std_2 = np.std(agmap_1), np.std(agmap_2)
        avg_mean_1 += mean_1
        avg_mean_2 += mean_2
        avg_std_1 += std_1
        avg_std_2 += std_2

        predicted_agmap = np.squeeze(output.cpu().numpy())
        agmap = np.squeeze(agmap.cpu().numpy())
        '''
        predicted_agmap[0, :, :] *= agmap_std[0]
        predicted_agmap[1, :, :] *= agmap_std[1]
        predicted_agmap[0, :, :] += agmap_mean[0]
        predicted_agmap[1, :, :] += agmap_mean[1]
        agmap[0, :, :] *= agmap_std[0]
        agmap[1, :, :] *= agmap_std[1]
        agmap[0, :, :] += agmap_mean[0]
        agmap[1, :, :] += agmap_mean[1]
        '''
        # predicted_agmap = cv2.resize(predicted_agmap, (agmap.shape[-1], agmap.shape[-2])) # (w, h)
        mse_loss += np.sum(np.square(predicted_agmap - agmap))
        np.save(os.path.join(predicted_agmap_dir, sample_batched['file_name'][0]), predicted_agmap)

    print('inference mse loss: {}'.format(mse_loss/len(dataloader)))
    print('mean_1: {}, mean_2: {}, std_1: {}, std_2: {}'.format(
        avg_mean_1/len(dataloader), avg_mean_2/len(dataloader), avg_std_1/len(dataloader), avg_std_2/len(dataloader)))


def inference_with_agmap(image_dir, ldet_dir, c5_dir, config_file, weight_file, gt=None):
    """
    读取原图像，计算粗分辨率检测结果，取出c5特征，计算agmap，分割，进行高分辨率推理，组合结果
    :param image_dir (str): 输入原始图像文件夹
    :param ldet_dir (str): 输入粗分辨率检测结果文件夹(spire格式json)
    :param c5_dir (str): 输入原始图像检测网络中的c5特征文件夹(npy格式)
    :return: None
    """

    image_fns = []
    fns = os.listdir(image_dir)
    fns.sort()

    # 加载cr模型
    checkpoint = torch.load(cr_saving_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNet(input_channels=2048).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    from spicv.spire_anno import SpireAnno
    from spicv.detection.structures.boxlist_ops import cat_boxlist
    from spicv.detection.structures.boxlist_ops import ignored_regions_iop

    spire_anno = SpireAnno(dataset='coco')
    detector = SpireDetector(config_file, weight_file, origin_size=True)

    for f in fns:
        if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png'):
            ldet_fn = os.path.join(ldet_dir, f) + '.json'
            ldet_boxlist = spire_anno.to_boxlist(ldet_fn)

            c5_fn = os.path.join(c5_dir, f) + '.npy'  ## 直接从硬盘中读取预存的c5特征
            c5_torch = torch.from_numpy(np.load(c5_fn)).to(device)
            with torch.no_grad():
                c5_torch = F.interpolate(c5_torch, size=c5_size, mode='bilinear', align_corners=False)
                output = model(c5_torch)
            predicted_agmap = np.squeeze(output.cpu().numpy())

            sub_wins, h_size = agmap_segmentation(predicted_agmap, ldet_boxlist.size, 800)
            image_fn = os.path.join(image_dir, f)
            image_fns.append(image_fn)
            image = cv2.imread(image_fn)
            image_h = image.copy()
            image_l = image.copy()

            image = cv2.resize(image, h_size)
            prediction_list = []
            for win in sub_wins:
                cv2.rectangle(image, (win[0], win[1]), (win[0]+win[2], win[1]+win[3]), (0,210,0), 2)
                image_win = image[win[1]:win[1]+win[3], win[0]:win[0]+win[2], :]
                prediction = detector.detect(image_win)
                prediction.bbox[:, 0] += win[0]
                prediction.bbox[:, 1] += win[1]
                prediction.bbox[:, 2] += win[0]
                prediction.bbox[:, 3] += win[1]
                prediction.size = h_size
                prediction = prediction.resize(ldet_boxlist.size)
                prediction_list.append(prediction)

            if len(sub_wins) > 0:
                sub_wins = BoxList(sub_wins, h_size, mode='xywh').resize(ldet_boxlist.size).convert(mode='xyxy')
                iou = ignored_regions_iop(sub_wins, ldet_boxlist, use_bbox=True)
                ldet_val, ldet_id = iou.max(dim=0)
                ldet_boxlist = ldet_boxlist[ldet_val < 0.5]

            prediction_list.append(ldet_boxlist)
            predictions = cat_boxlist(prediction_list)

            spire_anno.from_maskrcnn_benchmark(predictions, f, image.shape)

            image_show = spire_anno.visualize_boxlist(image, predictions, score_th=0.01)

            ## 以下只是为了显示
            hdet_fn = os.path.join(h_det_dir, f) + '.json'
            hdet_boxlist = spire_anno.to_boxlist(hdet_fn)
            image_h = spire_anno.visualize_boxlist(image_h, hdet_boxlist, score_th=0.01)
            cv2.imshow('image_h', image_h)

            ldet_fn = os.path.join(l_det_dir, f) + '.json'
            ldet_boxlist = spire_anno.to_boxlist(ldet_fn)
            image_l = spire_anno.visualize_boxlist(image_l, ldet_boxlist, score_th=0.01)
            cv2.imshow('image_l', image_l)

            agmap_fn = os.path.join(agmap_saving_dir_train, f) + '.npy'
            agmap = cv2.resize(np.load(agmap_fn), ldet_boxlist.size)
            agmap_color = agmap * alpha
            agmap_color = (agmap_color + 1) * 127
            agmap_color[agmap_color > 255] = 255
            agmap_color[agmap_color < 0] = 0
            agmap_color = cv2.applyColorMap(agmap_color.astype(np.uint8), cv2.COLORMAP_HOT)
            cv2.imshow("agmap", agmap_color)

            agmap_fn = os.path.join(predicted_agmap_dir, f) + '.npy'
            agmap = cv2.resize(np.load(agmap_fn), ldet_boxlist.size)
            agmap_color = (agmap + 1) * 127
            agmap_color[agmap_color > 255] = 255
            agmap_color[agmap_color < 0] = 0
            agmap_color = cv2.applyColorMap(agmap_color.astype(np.uint8), cv2.COLORMAP_HOT)
            cv2.imshow("prediected_agmap", agmap_color)

            cv2.imshow('image', image_show)
            cv2.waitKey(100)

            print(f)

    if gt is not None:
        eval_res = spire_anno.cocoapi_eval(gt)
    print('inference done!')


if __name__ == '__main__':
    spire_gt_dir_train = "/home/jario/dataset/coco/minval2014_spire_annotation/annotations"
    l_det_dir_train = "/home/jario/spire-net-1902/exps/fcos_r101_in300/annotations"
    # h_det_dir = "/home/jario/spire-net-1902/exps/fcos_r101_in800/annotations"

    agmap_saving_dir_train = '/media/jario/949AF0D79AF0B738/Experiments/fcos_agmap_cocominival'
    c5_saving_dir_train = '/media/jario/949AF0D79AF0B738/Experiments/fcos_f_cocominival/c5'

    cr_saving_fn = '/media/jario/949AF0D79AF0B738/Experiments/cr_model.pth.tar'
    predicted_agmap_dir = '/media/jario/949AF0D79AF0B738/Experiments/fcos_predicted_agmap'

    agmap_generate_test(agmap_saving_dir_train, spire_gt_dir_train, l_det_dir_train, show_agmap=False)
    cr_regressor_train(agmap_saving_dir_train, c5_saving_dir_train, cr_saving_fn)
    cr_regressor_inference(agmap_saving_dir_train, c5_saving_dir_train, cr_saving_fn, predicted_agmap_dir)

    image_dir = '/home/jario/dataset/coco/minival2014'
    config_file = '/home/jario/spire-net-1810/maskrcnn-benchmark-models/pre_trained/fcos_R_101_FPN_2x.yaml'
    weight_file = '/home/jario/spire-net-1810/maskrcnn-benchmark-models/pre_trained/FCOS_R_101_FPN_2x_s300.pth'
    gt = '/home/jario/dataset/coco/annotations/instances_minival2014.json'
    # inference_with_agmap(image_dir, l_det_dir, c5_saving_dir_train,
    #                      config_file, weight_file, gt=gt)

    print('complete!')
