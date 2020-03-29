# Spire Image Manager
Another toolset for image data annotation, preprocessing, visualization.

The labeling and visualization tools V4.1.3 (Windows x64) can be download on [**Baidu Pan**](https://pan.baidu.com/s/1Y8VdylCuk6q10XYe680FFA) (password: fdqq) or [**Spire Web**](http://121.36.68.10/tools/ImageLabelTools-4.1.3.zip).

Old versions:

V4.1.1 [**Baidu Pan**](https://pan.baidu.com/s/1_-QcF9hZrjE5S3-5WBfy7g) (password: 9gkl).

V4.0.5 [**Baidu Pan**](https://pan.baidu.com/s/1-fZyc5bvSzS4O3CQMxvTLA) (password: 7bqi).

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

