<div align="center">
  <img src="demo/spire-logo.png" width="400"/>
</div>

---

> **The git repository has been migrated to** https://gitee.com/jario-jin/SpireView.

Another toolset for image&video data annotation, preprocessing and visualization.

客户端当前版本：**v5.2.7**，下载地址：[**Baidu Pan (Windows x86-64)**](https://pan.baidu.com/s/1AlGJHmbeWVZXaBmL4P64SQ?pwd=3jjf).

1. 修复图片文件数量多于标注`json`文件数量时，`Ctrl+O`报错
2. 新增在播放`SpireCV`视频时，`Alt+R`可实现在线录制，方便试验成功视频的保存（PPT用^_^）
3. 新增`Tools->Terminal->gea4i`命令，针对负样本图片文件夹，自动生成空的`json`标注文件
4. 新增`Tools->StopWatch`，用于测量相机采集延迟
5. 优化标注对话框界面
6. 新增对DJI平台SRT文件信息的支持，see [dji_vid_srt_to_spire.py](https://gitee.com/jario-jin/SpireView/blob/master/to-spire-annotation/dji_vid_srt_to_spire.py)

支持[Segment-Anything-Model (SAM)](https://github.com/facebookresearch/segment-anything.git)，[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO.git)标注，部署服务端请参考：

1. [服务端本地部署方法](https://www.wolai.com/rGgq3Fm4SLGPcwFsjrX7ig)
2. [SpireView客户端与服务端通讯协议](https://www.wolai.com/f1BeT23zZp7rTjRL8aZNUD)
3. [B站视频教程](https://space.bilibili.com/516880777?spm_id_from=333.1007.0.0)

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

