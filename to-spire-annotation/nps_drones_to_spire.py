import cv2
import os
import json

#print("{{\"annos\":[{{\"area\":{},\"bbox\":[{},{},{},{}],\"category_name\":\"drone\"}}],\"file_name\":\"{}\",\"height\":{},\"width\":{}}}\n"
#    .format(2, 11, 22, 33, 44, ".jpg", 1080, 1920))


file_path = "C:/dataset/ups-drones"

txt_path = os.path.join(file_path, "NPS-Drones-Dataset")
video_path = os.path.join(file_path, "Videos")
image_path = os.path.join(file_path, "ups-drones-v211028/scaled_images")
annotations_path = os.path.join(file_path, "ups-drones-v211028/annotations")

n = 50


def find_all(str_in, target):
    result = []
    num = -1
    while 1:
        a = str_in[num+1:].find(target)
        if a != -1:
            num = num+a+1
            result.append(num)
        else:
            break

    return result


for i in range(n):
    # 读取视频
    # i=41
    dir_name = "211028-nps-drones-{}".format(str(i+1).zfill(2))
    video_name = "Clip_{}.mov".format(i+1)
    cap = cv2.VideoCapture(os.path.join(video_path, video_name))

    FrameNumber = int(cap.get(7))

    if not cap.isOpened():
        print("没有读取到视频文件...退出...")
        exit()
    else:
        print("读取视频: " + os.path.join(video_path, video_name) + "...")
    # 读取文件
    txt_name = "Clip_{}.txt".format(str(i+1).zfill(3))
    fi = open(os.path.join(txt_path, txt_name), 'r', encoding='utf-8')
    cnt = 0

    fid_from_txt = []
    nobj_from_txt = []
    xyxy_from_txt = []
    for line in fi:
        ll = line.strip().split(',')
        fid_from_txt.append(int(ll[0]))
        nobj_from_txt.append(int(ll[1]))
        boxes = []
        for obj_idx in range(int(ll[1])):
            boxes.append([float(ll[obj_idx*4 + 2]), float(ll[obj_idx*4 + 3]), float(ll[obj_idx*4 + 4]), float(ll[obj_idx*4 + 5])])
        xyxy_from_txt.append(boxes)
        cnt += 1

    print(fid_from_txt)
    print(nobj_from_txt)
    print(xyxy_from_txt)

    assert len(fid_from_txt) == len(nobj_from_txt)

    print(min(fid_from_txt))
    print(max(fid_from_txt))
    fid_min, fid_max = min(fid_from_txt), max(fid_from_txt)

    fi.close()
    # assert FrameNumber == cnt, "File line number != VideoFrameNumber"
    # fi = open(os.path.join(txt_path, txt_name), 'r', encoding='utf-8')
    # print("读取: txt" + os.path.join(txt_path, txt_name) + "...")
    cnt = -1

    full_image_path = os.path.join(image_path, dir_name)
    full_annotations_path = os.path.join(annotations_path, dir_name)

    if not os.path.exists(full_image_path):
        os.makedirs(full_image_path)
    if not os.path.exists(full_annotations_path):
        os.makedirs(full_annotations_path)

    cnt_index = dict()
    for index, fid in enumerate(fid_from_txt):
        cnt_index[fid] = index

    while cnt < fid_max:
        cnt += 1
        ret, frame = cap.read()
        if cnt in fid_from_txt:
            img_name = "Clip_{:0>2d}_{:0>6d}.jpg".format(i + 1, cnt)
            cv2.imwrite(os.path.join(full_image_path, img_name), frame)
            image_width = frame.shape[1]
            image_height = frame.shape[0]

            spire_dict = dict()
            spire_dict['file_name'] = img_name
            spire_dict['height'], spire_dict['width'] = image_height, image_width
            n_steps = 10
            frames_previous = []
            frame_bottom = max(fid_min, cnt - n_steps)
            for j in range(frame_bottom, cnt, 1):
                # assert j in fid_from_txt
                if j in fid_from_txt:
                    frames_previous.append("Clip_{:0>2d}_{:0>6d}.jpg".format(i + 1, j))
            frames_next = []
            frame_up = min(fid_max, cnt + n_steps + 1)
            for j in range(cnt + 1, frame_up, 1):
                # assert j in fid_from_txt, "j {}, fid_from_txt: {}".format(j, fid_from_txt)
                if j in fid_from_txt:
                    frames_next.append("Clip_{:0>2d}_{:0>6d}.jpg".format(i + 1, j))
                else:
                    break
            spire_dict['frames_previous'] = frames_previous
            spire_dict['frames_next'] = frames_next
            spire_dict['annos'] = []

            for j in range(nobj_from_txt[cnt_index[cnt]]):
                pos = xyxy_from_txt[cnt_index[cnt]][j]
                for k in range(4):
                    pos[k] = max(0, pos[k])
                pos[0] = min(image_width, pos[0])
                pos[2] = min(image_width, pos[2])
                pos[1] = min(image_height, pos[1])
                pos[3] = min(image_height, pos[3])

                # print(pos)
                target_width = pos[2] - pos[0]
                target_height = pos[3] - pos[1]
                assert target_height > 0 and target_width > 0

                target_area = target_width * target_height

                spire_anno = dict()
                spire_anno['area'] = target_area
                spire_anno['bbox'] = [pos[0], pos[1], target_width, target_height]
                spire_anno['category_name'] = 'drone'
                spire_dict['annos'].append(spire_anno)

            new_annos_name = img_name + ".json"
            with open(os.path.join(full_annotations_path, new_annos_name), "w") as f:
                json.dump(spire_dict, f)

    cap.release()
    cv2.destroyAllWindows()

print("转换完成！！！")
