<div align="center">
  <img src="demo/spire-logo.png" width="400"/>
</div>

---

> **The git repository has been migrated to** https://gitee.com/jario-jin/SpireView.

Another toolset for image&video data annotation, preprocessing and visualization.

客户端当前版本：**v5.3.0**，下载地址：[**Baidu Pan (Windows x86-64)**](https://pan.baidu.com/s/17DiQ9546AzJyy4WEdSR3yA?pwd=5mbg).

1. 支持新版SAM2、GroundingDINO自动化标注（无需自设服务器）
2. 修复BUG


支持[Segment-Anything-Model (SAM)](https://github.com/facebookresearch/segment-anything.git)，[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO.git)标注，如需要局域网部署，参考：

1. [服务端本地部署方法](https://www.wolai.com/5ADBx74qv13EP4SDW2tv4i)
2. [B站视频教程](https://space.bilibili.com/516880777?spm_id_from=333.1007.0.0)

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

