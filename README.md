# containers-v1-3d-bbox-to-pixels
Containers v1 - Get the 3D bboxes of items in renders and project them to pixel space

There are 3 unique SKUs. Each render contains objects of the same SKU, with different 
color schemes. The bounding box of each SKU at zero rotation/translation is hard-coded.
Based on the pose of each object in the info file, we calculate the axis-aligned 
3d bounding box of each object, then project them to pixels.

There is information about the renders in the info.json and metadata.jsonl files.
However, some information is missing. This data is hard-coded within the script:
1. The bounding box of each SKU at zero rotation/translation
2. Camera information (sensor size, field of view) 

10 Sample images are included in the repo.

![](sample_data/0.rgb.png)

## Usage

```shell script
python containers_3d_bbox_to_pixels.py dir_data=sample_data/
```

## Install
Requires Python 3.

```shell script
pip install -r requirements.txt
```
