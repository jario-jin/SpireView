import json
import os
import cv2
import argparse
from tqdm import tqdm



def main():
    parser = argparse.ArgumentParser(description="Convert txt formal annotation to spire annotation")
    parser.add_argument(
        "--Dota-dir",
        default="/home/bitvision/dataset/Dota/val/labelTxt",
        help="path to txt annotation file",
        # required=True
    )
    parser.add_argument(
        "--image-dir",
        default="/home/bitvision/dataset/Dota/val/images",
        help="path to image dir",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/bitvision/dataset/Dota/val/new_annotations",
        help="path to spire home dir",
    )
    parser.add_argument(
        '--coefficient',
        default=1,
        help='coefficient of resize'
    )
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, 'annotations')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    txtnames = os.listdir(args.Dota_dir)
    for txt in tqdm(txtnames):
        imagename = txt[:-4]+'.png'    #get the image name
        jsonname =  imagename+'.json'
        json_path = os.path.join(args.output_dir, jsonname)
        image_path = os.path.join(args.image_dir,imagename)
        img = cv2.imread(image_path)
        sp = img.shape
        with open(json_path, 'w')as jsonformal:
            annos = []             
            i = 0
            #print(oldname)
            filepath = os.path.join(args.Dota_dir,txt)
            with open(filepath,'r') as fp:
                f = fp.readlines()[2:]                              #start reading line at third
                for line in f:
                    data = line.split(" ")              
                    if(sp[0]<1200 or sp[1]<1200):
                        coefficient = args.coefficient
                    else:
                        if(sp[0]<sp[1]):
                            coefficient =args.coefficient    # float(1200/sp[0])
                        else:
                            coefficient =args.coefficient    # float(1200/sp[1])
                    X1,Y1,X2,Y2,X3,Y3,X4,Y4 = float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5]),float(data[6]),float(data[7])
                    X1, Y1, X2, Y2, X3, Y3, X4, Y4 = X1*coefficient, Y1*coefficient, X2*coefficient, Y2*coefficient, X3*coefficient, Y3*coefficient, X4*coefficient, Y4*coefficient
                    Xmin,Xmax = min(X1,X2,X3,X4),max(X1,X2,X3,X4)
                    Ymin,Ymax = min(Y1,Y2,Y3,Y4),max(Y1,Y2,Y3,Y4)
                    W = Xmax - Xmin
                    H = Ymax - Ymin
                    area = W*H
                    category_name = data[8]
                    dict1 = {
                        'area':area,
                        'bbox':[Xmin,Ymin,W,H],
                        'segmentation':[X1,Y1,X2,Y2,X3,Y3,X4,Y4],
                        'category_name':category_name
                        #'score':float(annos[length]['score'])

                    }
                    annos.append(dict1)
                dict1 = {
                    'annos':annos,
                    'file_name':imagename,
                    'height':sp[0],
                    'width':sp[1]
                }
                json.dump(dict1,jsonformal)

if __name__ == '__main__':
    main()
