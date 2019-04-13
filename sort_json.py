import json
import os
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
#from util import sort
import sort

def to_sort(root):
    '''
    converte spire_json to sort txt
    '''
    file_root = root
    annotations = os.listdir(file_root)
    result = []

    for frame,annotation in enumerate(annotations):
        line = []
        frame = frame + 1
        annotation_root = os.path.join(file_root,annotation)
        with open(annotation_root,'r') as fp:
            all = json.load(fp)
            annos = all['annos']
            for ann in annos:
                x,y,w,h,score = float(ann['bbox'][0]),float(ann['bbox'][1]),float(ann['bbox'][2]),float(ann['bbox'][3]),float(ann['score'])
                line = [frame, x, y, w, h, score]
                result.append(line)
                #print('%d,%.2f,%.2f,%.2f,%.2f,%2f' % (frame, x, y, w, h, score),
                    #  file=out_file)
    return result

def to_spire(spire_root,sorted_sipre_root,seq):
    annotations_root = spire_root  #'D:\\video\\annotations\\Pedestrian_5'
    sort_root = 'output\\'+seq+'.txt' #'C:\\Users\\DK\\Desktop\\目标检测\\sort-master\\output\\Pedestrian_5.txt'
    new_path = sorted_sipre_root #'D:\\video\\tracked_annotations\\Pedestrian_5'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    annotations = os.listdir(annotations_root)
    sort = np.loadtxt(sort_root, delimiter=',')[:, 0:6]
    for frame, anntation in enumerate(annotations):
        frame = frame + 1
        anntation_root = os.path.join(annotations_root, anntation)
        with open(anntation_root, 'r') as fp:
            all = json.load(fp)
            ids = sort[sort[:, 0] == frame, :]
            annos = all['annos']
            for ann in annos:
                x, y, w, h = round(ann['bbox'][0], 2), round(ann['bbox'][1], 2), round(ann['bbox'][2], 2), round(
                    ann['bbox'][3], 2)
                lenann = len(annos)
                lenids = len(ids)
                for id in ids:
                    iou = 0
                    x1, y1, w1, h1 = id[2], id[3], id[4], id[5]
                    x1max, y1max = max(x1, x), max(y1, y)
                    x2min, y2min = min(x1 + h1, x + h), min(y1 + w1, y + w)
                    if x1max < x2min:
                        area = (x2min - x1max) * (y2min - y1max)
                        iou = area / (w * h + w1 * h1 - area)
                    if (iou > 0.8):
                        ann['tracked_id'] = int(id[1])
                        break
            with open(os.path.join(new_path, anntation), 'w') as wp:
                json.dump(all, wp)
                print(anntation)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--spire-dir', default="D:\\need_sort",help="path to spire annotation file")
    parser.add_argument('--sorted-dir', default="D:\sort", help="path to sorted annotation file")
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    txt_root = ''
    args = parse_args()
    display = args.display
    phase = 'train'
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display

    if (display):
        if not os.path.exists('mot_benchmark'):
            print(
                '\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        plt.ion()
        fig = plt.figure()

    if not os.path.exists('output'):
        os.makedirs('output')
    annos_root = args.spire_dir
    sorted_root = args.sorted_dir
    sequences = os.listdir(annos_root)
    for seq in sequences:
        annos_list = to_sort(os.path.join(annos_root,seq))  # get the txt needed
        mot_tracker = sort.Sort()  # create instance of the SORT tracker
        #seq_dets = np.loadtxt(os.path.join(annos_root, seq), delimiter=',')  # load detections
        seq_dets = np.array(annos_list)
        with open('output/%s.txt' % (seq), 'w') as out_file:
            print("Processing %s." % (seq))
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 1:6]
                dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                if (display):
                    ax1 = fig.add_subplot(111, aspect='equal')
                    fn = 'mot_benchmark/%s/%s/img1/%06d.jpg' % (phase, seq, frame)
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')

                start_time = time.time()
                trackers = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                          file=out_file)
                    if (display):
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
                                                        ec=colours[d[4] % 32, :]))
                        ax1.set_adjustable('box-forced')

                if (display):
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()
        to_spire(os.path.join(annos_root,seq), os.path.join(sorted_root,seq), seq)

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))
    if (display):
        print("Note: to get real runtime results run without the option: --display")

