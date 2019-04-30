import os
import json
from tqdm import tqdm
import cv2
import argparse
import shutil

Car_1_list = [7, 27, 6, 31, 8, 18, 23, 2, 5, 9, 3, 14, 12, 186, 122, 556, 563, 20, 135, 427, 370, 206, 209, 221, 819,
              833, 1115]
Car_2_list = [22, 18, 50, 32, 17, 20, 254, 1224, 29, 130, 40, 264, 21, 2123, 16, 24, 1734, 2791, 3188, 29, 154, 565,
              2508, 1695, 909, 1041, 1185, 23, 36, 26, 33, 19]
Car_3_list = []
Car_4_list = [7, 6, 10, 49, 4, 8, 11, 35, 15, 13, 50, 19, 32, 229, 2, 91, 290, 231, 421, 504, 283, 69, 370, 793, 849,
              826, 926, 857, 878]
Car_5_list = [1360, 1311, 1500, 1497, 1465, 1613, 1375, 1697, 1710, 1215, 12, 34, 523, 2994, 1629, 980, 2179, 1360,
              2289, 2699, 1481, 1459, 1439, 1428, 1412, 1363, 1716, 1302, 1545, 1377, 1633, 1244, 1350, 1427, 2182,
              1651, 8, 2140, 2176, 2160, 2162, 2158, 2167, 1341, 599, 642, 585, 1579, 13, 2972]
Car_6_list = [1, 2, 3, 13, 38, 88, 91, 101, 110, 377, 372]
P_1_list = [10, 427, 525, 863, 773, 867, 887]
P_2_list = [29, 26, 27, 30, 66, 25, 681, 719, 777, 854, 1410, 1276, 1456, 1317]
P_3_list = [6, 9, 7, 11, 291, 251, 252, 807]
P_4_list = []
P_5_list = [360, 388, 871, 334, 347, 421, 390]
list_all = {'Car_1': Car_1_list, 'Car_2': Car_2_list, 'Car_3': Car_3_list, 'Car_4': Car_4_list, 'Car_5': Car_5_list,
            'P_1': P_1_list, 'P_2': P_2_list,
            'P_3': P_3_list, 'P_4': P_4_list, 'P_5': P_5_list, 'Car_6': Car_6_list}

def get_annotations(path,keyword):
    '''得到对应id号的annotations，分别放到对应的文件夹'''

    #得到对应的id好list
    for key,value in list_all.items():
        if keyword==key:
            num_list = value
            break
    root = path
    annotations_seqsroot = os.path.join(root,'annotations_sequence')   #各序列的annotations的路径
    if not os.path.exists(annotations_seqsroot):
        os.makedirs(annotations_seqsroot)
    anntationfile = os.path.join(root, 'annotations')     #所有annotations的路径
    for id in tqdm(num_list):
        annotations = os.listdir(anntationfile)
        seqname = str(keyword)+'_seq_'+ str(id)
        seqroot = os.path.join(annotations_seqsroot, seqname)       #单个seq的路径
        if not os.path.exists(seqroot):
            os.makedirs(seqroot)

        for anntation in annotations:
            annotationpath = os.path.join(anntationfile, anntation)
            with open(annotationpath, 'r') as fp:
                # print(anntation[0:20])
                annsall = json.load(fp)
                anns = annsall['annos']
                for ann in anns:
                    if ('tracked_id' in ann):
                        if (ann['tracked_id'] == id):
                            dict1 = {
                                'file_name': annsall['file_name'],
                                'height': annsall['height'],
                                'width': annsall['width'],
                                'annos': [ann]
                            }
                            savepath = os.path.join(seqroot, anntation)
                            with open(savepath, 'w') as dp:
                                json.dump(dict1, dp)
                            break
                        else:
                            continue
                    else:
                        continue

def get_txt(path):
    '''从annotations合并得到相应的txt格式的注释'''
    fileroot = os.path.join(path,'annotations_sequence')
    txtroot = os.path.join(path,'txt')
    if not os.path.exists(txtroot):
        os.makedirs(txtroot)



    Seqroots = os.listdir(fileroot)
    for Seqroot in tqdm(Seqroots):
        txtfile = os.path.join(txtroot, Seqroot + '.txt')
        root = os.path.join(fileroot, Seqroot)
        Sequences = os.listdir(root)
        with open(txtfile, 'w') as wp:
            for seq in Sequences:
                jsonfile = os.path.join(root, seq)
                with open(jsonfile,'r') as fp:
                    all = json.load(fp)
                    bbox = all['annos'][0]['bbox']
                    wp.write('%.f,%.f,%.f,%.f\n' % (bbox[0], bbox[1], bbox[2], bbox[3]))

def get_img(path, coe):
    '''得到相应的images的序列,并且缩小图片'''
    sequences_root = os.path.join(path,'annotations_sequence')
    image_root = os.path.join(path,'images')
    saveroot = os.path.join(path, 'images_sequence')
    if not os.path.exists(saveroot):
        os.makedirs(saveroot)

    sequences = os.listdir(sequences_root)
    for seq in tqdm(sequences):
        fileroot = os.path.join(sequences_root,seq)
        seqroot = os.path.join(saveroot,seq)
        if not os.path.exists(seqroot):
            os.makedirs(seqroot)
        annos = os.listdir(fileroot)
        for ann in annos:
            imagename = ann[0:ann.rfind('.')]
            imgroot = os.path.join(image_root,imagename)
            save = os.path.join(seqroot,imagename)
            img = cv2.imread(imgroot)
            h, w, c = img.shape
            img2 = cv2.resize(img, (int(w * coe), int(h * coe)))
            cv2.imwrite(save, img2)

def get_attr(path,keyword):
    '''得到visdrine中sot所需要的attributes'''
    root = os.path.join(path,'attri')
    for key,value in list_all.items():
        if keyword==key:
            num_list = value
            break
    for i in tqdm(num_list):
        name = str(keyword)+'_seq_'+str(i)+'_attr.txt'
        txt = os.path.join(root,name)
        with open(txt,'w') as fp:
            fp.write('0,0,0,0,0,0,0,0,0,0,0,0')

def delete(path):
    '''删除没有标注的多余的annotations'''
    annotations_root = os.path.join(path,'annotations_sequence')
    seqs = os.listdir(annotations_root)
    for seq in seqs:
        seqroot = os.path.join(annotations_root,seq)
        annotations = os.listdir(seqroot)
        for annotation in annotations:
            if int(annotation[7:11])>200:
                annroot = os.path.join(seqroot,annotation)
                os.remove(annroot)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Give spire root')
    parser.add_argument('--root', default="/home/bitvision/dataset/visdrone/Car1",help="path to spire annotation file")
    #arser.add_argument('--txt-dir', default="/home/bitvision/dataset/txt_visdrone/", help="path to visdrone txt file")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    root = parse_args().root
    #root为包含图片和注释的文件夹目录，图片命名为images，注释命名为annotations

    #get_annotations(path='root',keyword='P_3')
    ''':keyword  是指得到相应id号list的关键词'''

    #get_img(path='root',coe=0.2)
    ''':coe 指的是照片比例缩小的继续，为了节省储存空间 '''
    #get_txt(path=root)

    ##get_attr(path=root,keyword='P_3')
    ''':keyword  是指得到相应id号list的关键词'''

    #delete(root)


