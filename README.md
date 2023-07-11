<div align="center">
  <img src="demo/spire-logo.png" width="400"/>
</div>

---

> **The git repository has been migrated to** https://gitee.com/jario-jin/SpireView.

Another toolset for image&video data annotation, preprocessing and visualization.

当前版本：**v5.1.2**，下载地址：[**Baidu Pan (Windows x86-64)**](https://pan.baidu.com/s/1duqTr6AdrfNyXm8Sx7c9qw?pwd=4iub).

1. 界面优化吸附左侧，主要针对手工标注，矩形框、跟踪框、实例分割、关键点，增加右侧目标列表
2. 修复Bug，删减无用小功能

上一版本：**v5.1.1**，下载地址：[**Baidu Pan (Windows x86-64)**](https://pan.baidu.com/s/1piD3zLBx3TRl3yisQ-gEKw?pwd=qdra).

---

支持[Segment-Anything-Model (SAM)](https://github.com/facebookresearch/segment-anything.git)，本地部署方法如下：

```bash
# 1. 下载SpireView源代码（运行在Ubuntu上，用于提供SAM服务）
git clone https://gitee.com/jario-jin/SpireView.git
# 2. 安装SAM（环境：Ubuntu、CUDA、至少12G显存）
pip3 install opencv-python pycocotools matplotlib onnxruntime onnx
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip3 install -e .
# 3. 下载模型，转换ONNX
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
python3 scripts/export_onnx_model.py --checkpoint sam_vit_h_4b8939.pth --model-type vit_h --output sam_vit_h_4b8939.onnx --return-single-mask --opset 16
# 4. 拷贝模型到SpireView
cp sam_vit_h_4b8939.pth <path-to-SpireView>/SAM/
cp sam_vit_h_4b8939.onnx <path-to-SpireView>/SAM/
# 5. 运行本地SAM服务
cd <path-to-SpireView>/SAM/
python3 SAM_server.py
# 6. 打开另一个终端，查看Ubuntu本地IP地址（后面需要在Windows版的SpireView中输入）
ifconfig
# 7. 下载SpireView最新Windows版本软件，按B站视频教程操作
```

[B站视频教程](https://space.bilibili.com/516880777?spm_id_from=333.1007.0.0)

> 转换ONNX报错，请参考：https://github.com/facebookresearch/segment-anything/pull/210/Files

## How to use
![labeling](demo/labeling.jpg)

Support a variety of annotations.

<tr>
<td><img src="demo/bbox_labeling.gif" width="50%"></td>
<td><img src="demo/instance_labeling.gif" width="50%"></td>
</tr>

## Evaluation of spire annotations
see [EVALUATION_README.md](utils/evaluate/README.md)

## One json corresponds to an image
```bash
{
	"annos": [{
		"area": 277,
		"bbox": [552, 251, 24, 17],
		"category_name": "car",
		"segmentation": [[561, 253, 552, 263, 558, 266, 564, 268, 573, 266, 576, 260, 576, 254, 572, 251]]
	}],
	"file_name": "000009.jpg",
	"height": 720,
	"width": 1280
}
```

## Conversion between spire and MS COCO format
Convert MS COCO annotations to spire annotations.
```bash
python to-spire-annotation/coco_to_spire.py --coco-anno path_to_coco_json --coco-image-dir path_to_coco_image_dir --output-dir spire_annotation_dir
```

Convert spire annotations to MS COCO annotations.
![convert](demo/convert.png)

## Data statistics
![stat](demo/stat.png)

