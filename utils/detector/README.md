# Spire Image Manager
Install maskrcnn-benchmark (https://github.com/facebookresearch/maskrcnn-benchmark.git)

Modify the code
```
# maskrcnn_benchmark/maskrcnn_benchmark/structures/bounding_box.py
if bbox.ndimension() == 1 and len(bbox) == 4: # add these two lines
    bbox = bbox.unsqueeze(dim=0)              #

if bbox.ndimension() != 2:
    raise ValueError(
        "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
    )
```
