import json
import os
import sys
import logging
import pycocotools.mask as maskUtils
import numpy as np
import subprocess
import cv2


def load_class_desc(dataset='coco', logger=logging.getLogger()):
    """
    载入class_desc文件夹中的类别信息，txt文件的每一行代表一个类别
    :param dataset: str 'coco'
    :return: list ['cls1', 'cls2', ...]
    """
    desc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../detector/class_desc')
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
    def __init__(self, dataset='coco', spire_dir='/tmp',
                 max_det=500,
                 logger=logging.getLogger()):
        self.logger = logger
        self.spire_dir = spire_dir
        self.classes = load_class_desc(dataset, logger)

        self.class_id = {}
        for id, cls in enumerate(self.classes):
            self.class_id[cls] = id

        self.anno_dir = None
        self.anno_colors = self._create_colors(len(self.classes))
        self.num_classes = len(self.classes)
        self.max_det = max_det

    def _create_colors(self, len=1):
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len)]
        colors = [(int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255)) for rgba in colors]
        return colors

    def _assert_classes(self, len_input):
        """
        防止输入类别不等于预定义数据集类别
        :param len_input: int
        :return:
        """
        assert len_input == len(self.classes), '[spire]: Input length ({}) != len(self.classes)'.format(len_input)

    def _generate_coco_json(self, anno_dir=None):
        """
        根据spire annotations生成coco评价格式的json，之后调用cocoapi进行评价
        :param anno_dir (str): 外部输入的anno_dir，默认使用self.anno_dir
        :return:
        """
        if anno_dir is not None:
            self.anno_dir = anno_dir

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
                print(f)
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
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type, maxDet=self.max_det, saveEval=save_eval)
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
            self.anno_dir = anno_dir

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


class SpireFile:
    def __init__(self):
        self.json_dict = dict()

    def setup(self, file_name, height, width):
        """
        :param file_name: (str) e.g. name.jpg
        :param height: (int) e.g. 1080
        :param width: (int) e.g. 1920
        :return: None
        """
        self.json_dict['file_name'] = file_name
        self.json_dict['height'] = int(height)
        self.json_dict['width'] = int(width)
        self.json_dict['annos'] = []

    def add_annotation(self, bbox, score, category_name):
        """
        :param bbox: (float(4), nd.array) e.g. [x, y, width, height]
        :param score: (float) e.g. 0.85
        :param category_name: (str) e.g. "car"
        :return: None
        """
        anno = dict()
        anno['bbox'] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        anno['area'] = float(bbox[2] * bbox[3])
        anno['score'] = float(score)
        anno['category_name'] = category_name
        self.json_dict['annos'].append(anno)

    def output_file(self, output_path):
        """
        :param output_file_name: (str) e.g. "./annotations"
        :return: None
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(os.path.join(output_path, self.json_dict['file_name'] + ".json"), "w") as f:
            json.dump(self.json_dict, f)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    stream_handler = logging.StreamHandler(sys.stdout)
    fmt = '[%(asctime)s] [%(levelname)s] [ %(filename)s:%(lineno)s - %(name)s ] %(message)s '
    stream_handler.setFormatter(logging.Formatter(fmt))

    # logger.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)

    import argparse
    parser = argparse.ArgumentParser(description="Spire annotation demo")
    parser.add_argument(
        "--yolov4-dir",
        default='/home/jario/amov/darknet',
        help="yolov4 darknet root dir",
    )
    parser.add_argument(
        "--yolodata-file",
        default='/home/jario/amov/darknet/cfg/coco.data',
        help="path to yolo .data file",
    )
    parser.add_argument(
        "--yolocfg-file",
        default='/home/jario/amov/darknet/cfg/enet-coco.cfg',
        help="path to yolo .cfg file",
    )
    parser.add_argument(
        "--weights-file",
        default='/home/jario/amov/darknet/backup/enetb0-coco_final.weights',
        help="path to yolo .weights file",
    )
    parser.add_argument(
        "--test-images-dir",
        default='/home/jario/dataset/coco/minival2014',
        help="path to test images dir",
    )
    parser.add_argument(
        "--spire-dataset-name",
        default='coco',
        help="spire dataset name e.g. coco, mbzirc19_c1",
    )
    parser.add_argument(
        "--coco-gt",
        default='/home/jario/dataset/coco/annotations/instances_minival2014.json',
        help="coco gt",
    )
    args = parser.parse_args()

    yolov4_dir = args.yolov4_dir
    test_images_dir = args.test_images_dir
    image_fns = os.listdir(test_images_dir)

    i = 0
    f = open(os.path.join(yolov4_dir, 'jtest.txt'), 'w')
    for fn in image_fns:
        if fn.endswith(".jpg") or fn.endswith(".png") or fn.endswith(".jpeg") \
                or fn.endswith(".JPG") or fn.endswith(".JPEG"):
            file_path = os.path.join(test_images_dir, fn)
            f.write(file_path + '\n')
            # print(file_path)
            i += 1

    f.flush()
    f.close()
    print("TOTAL_IMAGES: {}".format(i))

    cmd = "cd {} && ./darknet detector test {} {} {} -thresh 0.01 -ext_output -dont_show " \
          "-out jresult.json < jtest.txt".format(yolov4_dir, args.yolodata_file,
        args.yolocfg_file, args.weights_file)
    status, output = subprocess.getstatusoutput(cmd)
    print(output)

    cls_names = load_class_desc(dataset=args.spire_dataset_name)
    print("CAT_NAMES: {}".format(cls_names))

    f = open(os.path.join(yolov4_dir, 'jresult.json'), encoding='utf-8')
    jresult = json.load(f)
    f.close()

    assert len(jresult) == i, "IMGAE NUM != JSON NUM"

    annotations_dir = os.path.join(yolov4_dir, 'annotations')
    if os.path.exists(annotations_dir):
        for root, dirs, files in os.walk(annotations_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.removedirs(annotations_dir)

    spf = SpireFile()
    for j in jresult:
        # print(j)
        file_path = j['filename']
        img = cv2.imread(file_path)
        path, name = os.path.split(file_path)
        h, w = img.shape[0], img.shape[1]
        spf.setup(name, h, w)
        for obj in j['objects']:
            ow = obj['relative_coordinates']['width'] * w
            oh = obj['relative_coordinates']['height'] * h
            spf.add_annotation([obj['relative_coordinates']['center_x'] * w - ow / 2.,
                                obj['relative_coordinates']['center_y'] * h - oh / 2.,
                                obj['relative_coordinates']['width'] * w,
                                obj['relative_coordinates']['height'] * h],
                               obj['confidence'], obj['name'])
        spf.output_file(annotations_dir)

    sa = SpireAnno(dataset=args.spire_dataset_name, spire_dir=annotations_dir, logger=logger)
    sa.cocoapi_eval(args.coco_gt, anno_dir=annotations_dir)
