# How to install dependencies
```
pip3 install numpy

git clone https://github.com/jario-jin/cocoapi.git
cd cocoapi/PythonAPI
python3 setup.py build_ext install
```

# How to run evaluation
```
cd **PATH_TO_spire-image-manager**/utils/evaluate
python3 spire_anno_eval.py --dataset mbzirc19_c1 --spire-dir PATH_TO_YOUR_SPIRE_ANNOTATION_DIR --gt PATH_TO_COCO_STYLE_GT
```
NOTE: PATH_TO_COCO_STYLE_GT: e.g. COCO_20190114_202242.json


$$\ell_{d}=\max \left[-\frac{1}{n} \sum_{i=1}^{n} \mathcal{L}_{d}^{i}(\mathbf{W}, \mathbf{b}, \mathbf{u}, z)-\frac{1}{n^{\prime}} \sum_{i=n+1}^{N} \mathcal{L}_{d}^{i}(\mathbf{W}, \mathbf{b}, \mathbf{u}, z)\right]$$

