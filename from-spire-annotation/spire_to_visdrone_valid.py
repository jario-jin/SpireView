#-*- coding: UTF-8 -*-
import json
import os
import argparse

class json_to_txt():
    def __init__(self,json_path=None,txt_path=None ):
        #ignored regions (0), pedestrian (1), people (2), bicycle (3), car (4), van (5), truck (6), tricycle (7), awning-tricycle (8), bus (9), motor (10), others (11)
        if not (json_path or txt_path):
            print('Path error')
        else:
            self.jsonpath = json_path
            self.txtpath = txt_path
    def tranform(self):
        filenames = os.listdir(self.jsonpath)
        category_name = {'ignored-regions':0, 'pedestrian':1,'people':2, 'bicycle':3, 'car' :4,
                         'van' :5,'truck' :6, 'tricycle': 7, 'awning-tricycle' :8, 'bus' :9, 'motor' :10, 'others ':11}
        for filename in filenames:
            filepath = os.path.join(self.jsonpath,filename)
            #print(filename)
            with open(filepath,'r') as fp:
                data = json.load(fp)
                annos = data['annos']
                file_name = self.txtpath+data['file_name']+'.txt'
                with open(file_name,'w') as txt:
                    lengths = len(annos)
                    for length in range(0,lengths):
                        x = str(annos[length]['bbox'][0])
                        y = str(annos[length]['bbox'][1])
                        width = str(annos[length]['bbox'][2])
                        height = str(annos[length]['bbox'][3])
                        category_index = str(annos[length]['category_name'])
                        #print(category_index)
                        for key,value in category_name.items():
                           if(category_index == key):
                               category = str(value)
                               continue
                        score = str(annos[length]['score'])
                        #score = str(annos[length]['score'])
                        write = x+','+y+','+width+','+height+','+score+','+category+',-1,-1'+'\n'
                        txt.write(write)
    print('Tranform have already done')

class BatchRename():
    '''
    batch rename the picture
    '''
    def __init__(self,txt_path=None):
        self.path =txt_path

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 1
        for item in filelist:
            #print(item)
            if item.endswith('.txt'):
                str = '.'
                new_name = item[0:23]
                #print(new_name)
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), new_name + '.txt')
                try:
                    os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print ('total %d to rename & converted %d jpgs' % (total_num, i-1))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--spire-dir', default="/home/bitvision/dataset/visdrone/",help="path to spire annotation file")
    parser.add_argument('--txt-dir', default="/home/bitvision/dataset/txt_visdrone/", help="path to visdrone txt file")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    txt,txtrename = json_to_txt(args.spire_dir,args.txt_dir),BatchRename(args.txt_dir)
    txt.tranform()
    txtrename.rename()