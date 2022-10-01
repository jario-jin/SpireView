import json
import os
import cv2
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Convert txt formal annotation to spire annotation")
    parser.add_argument(
        "--ann-txt-dir",
        default="C:/dataset/DOTA/train/labelTxt-v2.0/DOTA-v2.0_train",
        help="path to txt annotation file",
        # required=True
    )
    parser.add_argument(
        "--image-dir",
        default="C:/dataset/DOTA/train/images",
        help="path to image dir",
    )
    parser.add_argument(
        "--output-dir",
        default="C:/dataset/DOTA/BB211123-dota-train",
        help="path to spire home dir",
    )
    parser.add_argument(
        '--coefficient',
        default=1,
        help='coefficient of resize'
    )
    args = parser.parse_args()

    output_ann_dir = os.path.join(args.output_dir, 'annotations')
    if not os.path.exists(output_ann_dir):
        os.makedirs(output_ann_dir)

    scaled_images = os.path.join(args.output_dir, 'scaled_images')
    if not os.path.exists(scaled_images):
        os.makedirs(scaled_images)

    n_miss = []

    txtnames = os.listdir(args.ann_txt_dir)
    for i, txt in enumerate(txtnames):
        imagename = txt[:-4] + '.png'  # get the image name
        jsonname = imagename + '.json'
        output_json_path = os.path.join(output_ann_dir, jsonname)
        input_image_path = os.path.join(args.image_dir, imagename)
        print("{}, Image -- [{}]".format(str(i).zfill(6), input_image_path))
        img = cv2.imread(input_image_path)
        print(min(img.shape))

        # Prepare JSON dictionary for a single image.
        spire_dict = {}
        spire_dict['file_name'] = imagename
        img_h, img_w = img.shape[0], img.shape[1]
        spire_dict['height'], spire_dict['width'] = img_h, img_w
        spire_dict['annos'] = []

        with open(os.path.join(args.ann_txt_dir, txt), 'r') as fp:
            lines = fp.readlines()  # start reading line at third
            for line in lines:
                data = line.split(" ")
                X1, Y1, X2, Y2, X3, Y3, X4, Y4 = \
                    float(data[0]), float(data[1]), float(data[2]), float(data[3]), \
                    float(data[4]), float(data[5]), float(data[6]), float(data[7])
                category_name, diffcult = data[8], int(data[9])

                Xmin, Xmax = min(X1, X2, X3, X4), max(X1, X2, X3, X4)
                Ymin, Ymax = min(Y1, Y2, Y3, Y4), max(Y1, Y2, Y3, Y4)
                w = Xmax - Xmin
                h = Ymax - Ymin

                spire_anno = {}
                spire_anno['area'] = w * h
                spire_anno['bbox'] = [Xmin, Ymin, w, h]
                spire_anno['segmentation'] = [X1, Y1, X2, Y2, X3, Y3, X4, Y4]
                spire_anno['category_name'] = category_name
                spire_anno['obj_attrs'] = {'diffcult': diffcult}
                spire_dict['annos'].append(spire_anno)
                cv2.rectangle(img, (int(Xmin), int(Ymin)), (int(Xmax), int(Ymax)), (0, 0, 255), 1)

            try:
                cv2.imshow('img', img)
                cv2.waitKey(50)

                with open(output_json_path, "w") as f:
                    json.dump(spire_dict, f)

                open(os.path.join(scaled_images, imagename), 'wb').write(
                    open(input_image_path, 'rb').read())
            except Exception:
                n_miss.append(imagename)

    print(n_miss)
    print(len(n_miss))


if __name__ == '__main__':
    main()
