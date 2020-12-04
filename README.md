# containers-v1-3d-bbox-to-pixels
Containers v1 - Get the 3D bboxes of items in renders and project them to pixel space

There are 3 unique SKUs. Their bounding box at zero rotation/translation is hard-coded.
Based on the pose of each object in the info file, we calculate the 3d bounding box of
each object, then project them to pixels.

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
