#-*- coding: UTF-8 -*-
import json
import os

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
                           else:
                               print('Category error')
                        score = str(annos[length]['score'])
                        #score = str(annos[length]['score'])
                        write = x+','+y+','+width+','+height+','+score+','+category+',-1,-1'+'\n'
                        txt.write(write)
    print('Tranform have already done')

class BatchRename():
    '''
    批量重命名文件夹中的图片文件
    '''
    def __init__(self,txt_path=None):
        self.path =txt_path

    def rename(self):
        filelist = os.listdir(self.path) #获取文件路径
        total_num = len(filelist) #获取文件长度（个数）
        i = 1  #表示文件的命名是从1开始的
        for item in filelist:
            #print(item)
            if item.endswith('.txt'):  #初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
                str = '.'
                new_name = item[0:23]
                #print(new_name)
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), new_name + '.txt')
                #dst = os.path.join(os.path.abspath(self.path), 'afternoon2_'+format(str(i),'0>5') + '.jpg')#处理后的格式也为jpg格式的，当然这里可以改成png格式
                #dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')    这种情况下的命名格式为0000000.jpg形式，可以自主定义想要的格式
                try:
                    os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print ('total %d to rename & converted %d jpgs' % (total_num, i-1))

if __name__ == '__main__':
    json_path = ''
    txt_path = ''
    txt,txtrename = json_to_txt(json_path,txt_path),BatchRename(txt_path)
    txt.tranform()
    txtrename.rename()