# How to install dependencies
```
pip3 install numpy

git clone https://github.com/jario-jin/cocoapi.git
cd cocoapi/PythonAPI
sudo python3 setup.py build_ext --inplace
```

# How to run evaluation
```
cd **PATH_TO_spire-image-manager**/utils/evaluate
python3 spire_anno_eval.py --dataset mbzirc19_c1 --spire-dir PATH_TO_YOUR_SPIRE_ANNOTATION_DIR --gt PATH_TO_COCO_STYLE_GT
```
NOTE: PATH_TO_COCO_STYLE_GT: e.g. COCO_20190114_202242.json
