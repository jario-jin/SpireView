import json
import os
import sys
import logging
import pycocotools.mask as maskUtils
import numpy as np
import cv2


def find_contours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4
    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy


def load_class_desc(dataset='coco', logger=logging.getLogger()):
    """
    载入class_desc文件夹中的类别信息，txt文件的每一行代表一个类别
    :param dataset: str 'coco'
    :return: list ['cls1', 'cls2', ...]
    """
    desc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_desc')
    desc_names = []
    for f in os.listdir(desc_dir):
        if f.endswith('.txt'):
            desc_names.append(os.path.splitext(f)[0])
    # 如果类别描述文件存在，则返回所有类别名称，否则会报错
    cls_names = []
    if dataset in desc_names:
        with open(os.path.join(desc_dir, dataset + '.txt')) as f:
            for line in f.readlines():
                if len(line.strip()) > 0:
                    cls_names.append(line.strip())
    else:
        raise NameError('[spire]: {}.txt not exist in "class_desc"'.format(dataset))
    # 类别描述文件不能为空，否则会报错
    if len(cls_names) > 0:
        logger.info('loading {} class descriptions.'.format(len(cls_names)))
        return cls_names
    else:
        raise RuntimeError('[spire]: {}.txt is EMPTY'.format(dataset))


class SpireAnno(object):
    """
    spire格式标注生成器类
    """
    def __init__(self, dataset='coco', spire_dir='/tmp', class_transfor=None, logger=logging.getLogger()):
        self.logger = logger
        self.spire_dir = spire_dir
        self.classes = load_class_desc(dataset, logger)

        self.class_transfor = None
        self.class_id = {}
        for id, cls in enumerate(self.classes):
            self.class_id[cls] = id

        self.anno_dir = None
        self.anno_colors = self._create_colors(len(self.classes))
        self.num_classes = len(self.classes)

    def _create_colors(self, len=1):
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len)]
        colors = [(int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255)) for rgba in colors]
        return colors

    def to_boxlist(self, spire_file_name):
        """
        将spire格式的json文件转换为BoxList格式
        :param spire_file_name (str): e.g. 'COCO_val2014_000000000139.jpg.json'
                               (dict): {'file_name':'', 'height':...}
        :return:
        """
        if isinstance(spire_file_name, dict):
            ann_dict = spire_file_name
        else:
            ann_dict = json.loads(open(spire_file_name, 'r').read())

        assert 'annos' in ann_dict.keys(), "'annos' should be in spire_annotation_dict.keys()"
        # assert 'annos' in ann_dict.keys() and len(ann_dict['annos']) > 0, "len(ann_dict['annos']) should be > 0"

        heigth, width = ann_dict['height'], ann_dict['width']
        if len(ann_dict['annos']) == 0:
            bbox = np.zeros((0, 4), dtype=np.float32)
        else:
            bbox = np.array([b['bbox'] for b in ann_dict['annos']], dtype=np.float32)

        from spicv.detection.structures.bounding_box import BoxList
        import torch

        boxlist = BoxList(bbox, (width, heigth), mode="xywh").convert("xyxy")
        if len(ann_dict['annos']) > 0:
            ## 将空格替换为下划线
            labels = np.array([self.class_id[b['category_name'].replace(' ', '_')] \
                               for b in ann_dict['annos']], np.int64)
            boxlist.add_field("labels", torch.from_numpy(labels + 1))  # +1 表示加上背景类

        if len(ann_dict['annos']) > 0 and 'score' in ann_dict['annos'][0]:
            scores = np.array([b['score'] for b in ann_dict['annos']], np.float32)
            boxlist.add_field("scores", torch.from_numpy(scores))

        return boxlist

    def visualize_boxlist(self, image, boxlist, score_th=0.2):
        """
        将BoxList的检测结果显示在图像上
        :param image (np.ndarray): 输入图像opencv读取
        :param boxlist (BoxList): BoxList格式的检测结果
        :param score_th (float): 得分过滤
        :return:
        """
        nh, nw = image.shape[:2]

        import torch
        # reshape prediction (a BoxList) into the original image size
        boxlist = boxlist.resize((nw, nh)).convert("xywh")
        ##  sort by scores, and draw bbox
        scores = boxlist.get_field("scores")
        keep = torch.nonzero(scores > score_th).squeeze(1)
        boxlist = boxlist[keep]
        scores = boxlist.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        boxlist = boxlist[idx]

        ## coco_eval.py/prepare_for_coco_detection
        scores = boxlist.get_field("scores").tolist()
        labels = boxlist.get_field("labels").tolist()
        bboxes = boxlist.bbox.numpy().astype(np.int32).tolist()

        canvas = image.copy()
        coco_names = self.classes
        coco_colors = self.anno_colors
        for bbox, score, label in zip(bboxes, scores, labels):
            x, y, w, h = bbox
            label = label - 1 # -1 表示去掉背景类
            if label < len(coco_names):
                color = coco_colors[label]
                name = coco_names[label]
                caption = "#{} {} {:.3f}".format(label, name, score)
            else:
                color = (0, 255, 0)
                caption = "#{}, s({:.3f})".format(label, score)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
            cv2.putText(canvas, caption, (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1,
                        cv2.LINE_AA)
        return canvas

    def _assert_classes(self, len_input):
        """
        防止输入类别不等于预定义数据集类别
        :param len_input: int
        :return:
        """
        assert len_input == len(self.classes), '[spire]: Input length ({}) != len(self.classes)'.format(len_input)

    def from_maskrcnn_benchmark(self, result, image_name, image_size, confidence=0.01):
        """
        将maskrcnn-benchmark的结果转换为spire标注
        :param result (BoxList): |- bbox (Tensor) [n, 4]
                                 |- extra_fields (dict): |- scores (Tensor) [n,]
                                                         |- labels (Tensor-int64) [n,]
                                                         |- mask (Tensor) [n,1,28,28]
        :param image_name (str)
        :param image_size (tuple) - (height, width)
        :return:
        """
        import torch
        im_h, im_w = image_size[0], image_size[1]
        mask_np, mask = None, None

        bbox_np = result.bbox.cpu().numpy()
        scores_np = result.extra_fields['scores'].cpu().numpy()
        labels_np = result.extra_fields['labels'].cpu().numpy()
        if result.has_field('mask'):   # 有分割标注
            mask_np = result.extra_fields['mask'].cpu().numpy()

        image_name = os.path.basename(image_name)
        self.logger.debug("image_name: {}".format(image_name))

        annos = []
        for i in range(len(bbox_np)):
            bbox = bbox_np[i]
            score = scores_np[i]
            label = labels_np[i] - 1   # -1 表示去掉背景类

            class_ci = self.classes[label]
            if self.class_transfor is not None:
                class_ci = self.class_transfor[class_ci]
                if class_ci == '__':   # 需要丢掉的类别
                    continue

            if mask_np is not None:    # 有分割标注
                mask = mask_np[i]

            anno = {}
            TO_REMOVE = 1
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0] + TO_REMOVE, bbox[3] - bbox[1] + TO_REMOVE
            anno['bbox'] = [float(x), float(y), float(w), float(h)]
            anno['score'] = float(score)
            if anno['score'] < confidence:
                continue
            anno['category_name'] = class_ci
            anno['area'] = w * h
            if mask is not None:       # 有分割标注
                mask = mask[0]
                # 需要检查以下代码
                x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
                if x + w > im_w: w = im_w - x
                if y + h > im_h: h = im_h - y
                mask = cv2.resize(mask, (w, h))
                mask_map = np.zeros((im_h, im_w), np.uint8)
                mask_bin = np.zeros((h, w), np.uint8)
                mask_bin[mask > 0.5] = 255
                mask_map[y:y + h, x:x + w] = mask_bin
                contours, hierarchy = find_contours(mask_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                area = 0
                for contour in contours:
                    area += int(round(cv2.contourArea(contour)))
                if area != 0:
                    segmentations = []
                    for contour in contours:
                        segmentation = []
                        for cp in range(contour.shape[0]):
                            segmentation.append(int(contour[cp, 0, 0]))
                            segmentation.append(int(contour[cp, 0, 1]))
                        segmentations.append(segmentation)
                    anno['area'], anno['segmentation'] = area, segmentations
                else:
                    anno['area'], anno['segmentation'] = 0, []
            annos.append(anno)

        json_dict = {}
        json_dict['file_name'] = image_name
        json_dict['height'] = image_size[0]
        json_dict['width'] = image_size[1]
        json_dict['annos'] = annos

        anno_dir = os.path.join(self.spire_dir, 'annotations')
        if not os.path.exists(anno_dir):
            os.makedirs(anno_dir)

        self.anno_dir = anno_dir  # 保存anno_dir，用来合成测试json
        anno_fn = os.path.join(anno_dir, image_name + '.json')
        f = open(anno_fn, 'w', encoding='utf-8')
        json.dump(json_dict, f)

    def _generate_coco_json(self, anno_dir=None):
        """
        根据spire annotations生成coco评价格式的json，之后调用cocoapi进行评价
        :param anno_dir (str): 外部输入的anno_dir，默认使用self.anno_dir
        :return:
        """
        if anno_dir is not None:
            self.anno_dir = os.path.join(anno_dir, 'annotations')

        coco_results = []
        for image_id, mapped_id in self.id_to_img_map.items():
            file_name = self.coco.imgs[mapped_id]['file_name'] + '.json'
            '''
            name = self.coco.imgs[mapped_id]['file_name']
            import shutil
            p1 = os.path.join('/home/jario/dataset/coco/val2014', name)
            p2 = os.path.join('/home/jario/dataset/coco/minival2014', name)
            shutil.copy(p1, p2)
            '''
            with open(os.path.join(self.anno_dir, file_name), 'r') as f:
                json_str = f.read()
                json_dict = json.loads(json_str)

            image_width = self.coco.imgs[mapped_id]["width"]
            image_height = self.coco.imgs[mapped_id]["height"]

            boxes, scores, labels = [], [], []
            for anno in json_dict['annos']:
                boxes.append(anno['bbox'])
                scores.append(anno['score'])
                labels.append(self.class_id[anno['category_name']] + 1)

            mapped_labels = [self.contiguous_category_id_to_json_id[i] for i in labels]

            coco_results.extend(
            [
                {
                    "image_id": mapped_id,
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ])
        return coco_results


    def _eval_on_cocoapi(self, coco_gt, coco_results, json_result_file, iou_type="bbox", save_eval=''):
        """
        调用cocoapi进行实际评价
        :param coco_gt (cocoapi): COCO(coco_annotations)
        :param coco_results (dict): 自己生成的测试结果
        :param json_result_file (str): 将coco_results转换到json文件中
        :param iou_type (str): 评价类型['bbox', 'segm']
        :param save_eval (str): 是否输出cocoapi产生的中间结果
        :return:
        """
        with open(json_result_file, "w") as f:
            json.dump(coco_results, f)

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()

        # coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type, maxDet=500, saveEval=save_eval)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize(self.logger)
        return coco_eval


    def cocoapi_eval(self, ground_truth_json, anno_dir=None):
        """
        用cocoapi进行评价
        :param ground_truth_json (str): path to coco-format annotations, e.g. 'instances_minival2014.json'
        :param anno_dir (str): 外部输入的anno_dir，默认使用self.anno_dir
        :return:
        """
        from pycocotools.coco import COCO

        if anno_dir is not None:
            self.anno_dir = os.path.join(anno_dir, 'annotations')

        self.coco = COCO(ground_truth_json)
        self.ids = list(self.coco.imgs.keys())
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        coco_results_bbox = self._generate_coco_json(anno_dir)

        iou_type = 'bbox'
        return self._eval_on_cocoapi(
            self.coco,
            coco_results_bbox,
            os.path.join(self.spire_dir, iou_type + '.json'),
            iou_type,
            self.spire_dir
        )


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    stream_handler = logging.StreamHandler(sys.stdout)
    fmt = '[%(asctime)s] [%(levelname)s] [ %(filename)s:%(lineno)s - %(name)s ] %(message)s '
    stream_handler.setFormatter(logging.Formatter(fmt))

    logger.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)

    import argparse
    parser = argparse.ArgumentParser(description="Spire annotation demo")
    parser.add_argument(
        "--dataset",
        required=True,
        help="dataset name, can be choosen in ['coco','visdrone',...]",
    )
    parser.add_argument(
        "--spire-dir",
        required=True,
        help="path to spire annotation dir",
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="path to ground truth json file",
    )
    args = parser.parse_args()

    spire_dir = args.spire_dir
    if spire_dir.endswith('/'):
        spire_dir = spire_dir[:-1]
    if spire_dir.endswith('annotations'):
        spire_dir = os.path.dirname(spire_dir)

    sa = SpireAnno(dataset=args.dataset, spire_dir=spire_dir, logger=logger)
    sa.cocoapi_eval(args.gt, anno_dir=spire_dir)
