
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
        "--visdrone-mot-anno",
        default="D:/Dataset/VisDrone2019-MOT-val/annotations",
        help="path to VisDrone MOT annotation file",
    )
    parser.add_argument(
        "--visdrone-sequences-dir",
        default="D:/Dataset/VisDrone2019-MOT-val/sequences",
        help="path to VisDrone MOT sequences dir",
    )
    parser.add_argument(
        "--output-dir",
        default="D:/Dataset/spire-visdrone19-mot-val",
        help="path to spire home dir",
    )
    args = parser.parse_args()

    scaled_images = os.path.join(args.output_dir, 'scaled_images')
    args.output_dir = os.path.join(args.output_dir, 'annotations')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(scaled_images):
        os.makedirs(scaled_images)

    category_id_to_name = ['ignored-regions',
                           'pedestrian',
                           'people',
                           'bicycle',
                           'car',
                           'van',
                           'truck',
                           'tricycle',
                           'awning-tricycle',
                           'bus',
                           'motor',
                           'others']

    txt_names = os.listdir(args.visdrone_mot_anno)
    for txt in tqdm(txt_names):
        if txt[-4:] == '.txt':
            image_seq_dir = os.path.join(args.visdrone_sequences_dir, txt[:-4])

            txt_fn = os.path.join(args.visdrone_mot_anno, txt)
            f = open(txt_fn, 'r')
            anno_lines = f.readlines()
            """
            <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
            <object_category>:	  
                The object category indicates the type of annotated object, (i.e., ignored regions(0), pedestrian(1), 
                people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10), 
                others(11))
            """
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

            for fi in anno_frame_index.keys():
                image_name = str(fi).zfill(7) + '.jpg'
                image_fn = os.path.join(image_seq_dir, image_name)
                image = cv2.imread(image_fn)
                # cv2.imshow('image', image)
                # cv2.waitKey()
                # Prepare JSON dictionary for a single image.
                spire_dict = {}
                spire_dict['file_name'] = txt[:-4] + '_' + image_name
                spire_dict['height'], spire_dict['width'] = image.shape[0], image.shape[1]
                n_steps = 10
                frames_previous = []
                frame_bottom = max(frame_min, fi-n_steps)
                for i in range(frame_bottom, fi, 1):
                    frames_previous.append(txt[:-4] + '_' + str(i).zfill(7) + '.jpg')
                frames_next = []
                frame_up = min(frame_max+1, fi+n_steps+1)
                for i in range(fi+1, frame_up, 1):
                    frames_next.append(txt[:-4] + '_' + str(i).zfill(7) + '.jpg')
                spire_dict['frames_previous'] = frames_previous
                spire_dict['frames_next'] = frames_next
                spire_dict['annos'] = []
                for fi_ext in anno_frame_index[fi]:
                    target_id, bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, \
                    occlusion = int(fi_ext[0]), fi_ext[1], fi_ext[2], fi_ext[3], fi_ext[4], \
                                fi_ext[5], int(fi_ext[6]), int(fi_ext[7]), int(fi_ext[8])
                    spire_anno = {}
                    spire_anno['tracked_id'] = target_id
                    spire_anno['area'] = bbox_width * bbox_height
                    spire_anno['bbox'] = [bbox_left, bbox_top, bbox_width, bbox_height]
                    spire_anno['truncation'] = truncation
                    spire_anno['occlusion'] = occlusion
                    spire_anno['category_name'] = category_id_to_name[object_category]
                    spire_dict['annos'].append(spire_anno)

                # Generate spire annotation files for each image
                output_fn = os.path.join(args.output_dir, txt[:-4] + '_' + image_name + '.json')
                with open(output_fn, "w") as f:
                    json.dump(spire_dict, f)

                # seq_scaled_images = os.path.join(scaled_images, txt[:-4])
                # if not os.path.exists(seq_scaled_images):
                #     os.makedirs(seq_scaled_images)

                open(os.path.join(scaled_images, txt[:-4] + '_' + image_name), 'wb').write(
                    open(image_fn, 'rb').read())


if __name__ == '__main__':
    main()
