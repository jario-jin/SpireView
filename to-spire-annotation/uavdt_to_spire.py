
import argparse
import os
import json
import cv2
from tqdm import tqdm
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


def solve_coco_segs(segs_anno, h, w):
    assert type(segs_anno['counts']) == list, "segs_anno['counts'] should be list"
    rle = maskUtils.frPyObjects(segs_anno, h, w)
    mask = maskUtils.decode(rle)
    contours, hierarchy = find_contours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    for contour in contours:
        area += int(round(cv2.contourArea(contour)))
    if area != 0:
        segmentations = []
        for contour in contours:
            if contour.shape[0] >= 6:  # three points
                segmentation = []
                for cp in range(contour.shape[0]):
                    segmentation.append(int(contour[cp, 0, 0]))
                    segmentation.append(int(contour[cp, 0, 1]))
                segmentations.append(segmentation)
    else:
        segmentations = []
    return segmentations


def main():
    parser = argparse.ArgumentParser(description="Convert VisDrone annotation to spire annotation")
    parser.add_argument(
        "--uavdt-attr",
        default="D:/Dataset/M_attr/test",
        help="path to UAVDT attr file",
    )
    parser.add_argument(
        "--uavdt-anno",
        default="D:/Dataset/UAV-benchmark-MOTD_v1.0/GT",
        help="path to UAVDT annotation file",
    )
    parser.add_argument(
        "--uavdt-image-dir",
        default="D:/Dataset/UAV-benchmark-M",
        help="path to UAVDT image dir",
    )
    parser.add_argument(
        "--output-dir",
        default="D:/Dataset/MOT210426-UAVDT-test",
        help="path to spire home dir",
    )
    args = parser.parse_args()

    scaled_images = os.path.join(args.output_dir, 'scaled_images')
    args.output_dir = os.path.join(args.output_dir, 'annotations')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(scaled_images):
        os.makedirs(scaled_images)

    # car(1), truck(2), bus(3)
    category_id_to_name = ['car',
                           'truck',
                           'bus']

    txt_names = os.listdir(args.uavdt_attr)
    for txt in tqdm(txt_names):
        if txt[-4:] == '.txt':
            image_seq_dir = os.path.join(args.uavdt_image_dir, txt[:-9])
            # image = cv2.imread(image_fn)
            txt_fn = os.path.join(args.uavdt_attr, txt)
            f = open(txt_fn, 'r')
            anno_line = f.readlines()[0]
            f.close()
            # daylight, night, fog; low-alt, medium-alt, high-alt; front-view, side-view, bird-view; long-term.
            daylight, night, fog, low_alt, medium_alt, high_alt, front_view, side_view, bird_view, long_term = \
                anno_line.split(',')
            daylight, night, fog, low_alt, medium_alt, high_alt, front_view, side_view, bird_view, long_term = \
                int(daylight), int(night), int(fog), int(low_alt), int(medium_alt), int(high_alt), int(front_view), \
                int(side_view), int(bird_view), int(long_term)

            txt_fn = os.path.join(args.uavdt_anno, txt[:-8] + 'gt_whole.txt')
            f = open(txt_fn, 'r')
            anno_lines = f.readlines()
            f.close()
            # <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<out-of-view>,<occlusion>,<object_category>
            anno_frame_index = {}
            frame_min, frame_max = 1e8, -1
            for line in anno_lines:
                line_split = line.split(',')
                fi = int(line_split[0])
                fi_ext = [float(x) for x in line_split[1:]]
                if fi in anno_frame_index.keys():
                    anno_frame_index[fi].append(fi_ext)
                else:
                    anno_frame_index[fi] = [fi_ext]
                if fi > frame_max:
                    frame_max = fi
                if fi < frame_min:
                    frame_min = fi

            txti_fn = os.path.join(args.uavdt_anno, txt[:-8] + 'gt_ignore.txt')
            f = open(txti_fn, 'r')
            annoi_lines = f.readlines()
            f.close()
            for line in annoi_lines:
                line_split = line.split(',')
                fi = int(line_split[0])
                fi_ext = [float(x) for x in line_split[1:]]
                if fi in anno_frame_index.keys():
                    anno_frame_index[fi].append(fi_ext)
                else:
                    anno_frame_index[fi] = [fi_ext]
                if fi > frame_max:
                    frame_max = fi
                if fi < frame_min:
                    frame_min = fi

            for fi in anno_frame_index.keys():
                image_name = 'img' + str(fi).zfill(6) + '.jpg'
                image_fn = os.path.join(image_seq_dir, image_name)
                image = cv2.imread(image_fn)
                # cv2.imshow('image', image)
                # cv2.waitKey()
                # Prepare JSON dictionary for a single image.
                spire_dict = {}
                spire_dict['file_name'] = txt[:-9] + '_' + image_name
                spire_dict['height'], spire_dict['width'] = image.shape[0], image.shape[1]
                spire_dict['daylight'] = daylight
                spire_dict['night'] = night
                spire_dict['fog'] = fog
                spire_dict['low_alt'] = low_alt
                spire_dict['medium_alt'] = medium_alt
                spire_dict['high_alt'] = high_alt
                spire_dict['front_view'] = front_view
                spire_dict['side_view'] = side_view
                spire_dict['bird_view'] = bird_view
                spire_dict['long_term'] = long_term
                img_attrs = {"daylight": daylight,
                             "night": night,
                             "fog": fog,
                             "low-alt": low_alt,
                             "medium-alt": medium_alt,
                             "high-alt": high_alt,
                             "front-view": front_view,
                             "side-view": side_view,
                             "bird-view": bird_view}
                spire_dict['img_attrs'] = img_attrs
                n_steps = 10
                frames_previous = []
                frame_bottom = max(frame_min, fi - n_steps)
                for i in range(frame_bottom, fi, 1):
                    frames_previous.append(txt[:-9] + '_' + 'img' + str(i).zfill(6) + '.jpg')
                frames_next = []
                frame_up = min(frame_max + 1, fi + n_steps + 1)
                for i in range(fi + 1, frame_up, 1):
                    frames_next.append(txt[:-9] + '_' + 'img' + str(i).zfill(6) + '.jpg')
                spire_dict['frames_previous'] = frames_previous
                spire_dict['frames_next'] = frames_next
                spire_dict['annos'] = []
                for fi_ext in anno_frame_index[fi]:
                    target_id, bbox_left, bbox_top, bbox_width, bbox_height, out_of_view, occlusion, object_category \
                        = int(fi_ext[0]), fi_ext[1], fi_ext[2], fi_ext[3], fi_ext[4], \
                          int(fi_ext[5]), int(fi_ext[6]), int(fi_ext[7])
                    spire_anno = {}
                    spire_anno['tracked_id'] = target_id
                    spire_anno['area'] = bbox_width * bbox_height
                    spire_anno['bbox'] = [bbox_left, bbox_top, bbox_width, bbox_height]
                    spire_anno['out_of_view'] = 1 if out_of_view > 1 else 0
                    spire_anno['occlusion'] = 1 if occlusion > 1 else 0
                    if -1 == object_category:
                        spire_anno['category_name'] = 'ignored-regions'
                    else:
                        spire_anno['category_name'] = category_id_to_name[object_category - 1]

                    spire_anno['obj_attrs'] = {'occlusion': 1 if occlusion > 1 else 0,
                                               'truncation': 1 if out_of_view > 1 else 0,
                                               'invisibility': 0}
                    spire_dict['annos'].append(spire_anno)

                # Generate spire annotation files for each image
                output_fn = os.path.join(args.output_dir, txt[:-9] + '_' + image_name + '.json')
                with open(output_fn, "w") as f:
                    json.dump(spire_dict, f)

                open(os.path.join(scaled_images, txt[:-9] + '_' + image_name), 'wb').write(
                    open(image_fn, 'rb').read())


if __name__ == '__main__':
    main()
