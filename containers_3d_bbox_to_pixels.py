import jsonlines
from pathlib import Path
from typing import List, Union
import numpy as np
import math
import cv2
import json
from typing import Dict
from scipy.spatial.transform import Rotation as R
import random
import hydra
from omegaconf import OmegaConf, DictConfig


EXT_INFO = '.info.json'
EXT_RGB = '.rgb.png'
EXT_SEGMENTS = '.segments.png'
EXT_VIZ_BBOX = '.bbox.png'
FNAME_METADATA = 'metadata.jsonl'
SEED = 0
TEST_OBJ_ID = 192  # If not none, will only process this one obj in image, for debugging purposes. Obj 192 for renderid 0 is on top of pile.


# Data about the camera. Some of it is hardcoded and not available in metadata file
CAM_DATA = {
    # These are available in the metadata
    "resolution": {
        "w": 1024,  # Image width
        "h": 1024,  # Image height
    },
    "focal_len_mm": 32,

    # These are constants used for this job, not captured in metadata
    "sensor": {
        "width_mm": 41.4214,  # 33.0,
        "height_mm": 41.4214,  # 33.0
    },

    # These are unknown and must be calculated
    "field_of_view": {
        "y_axis_rads": 1.8260405563703,  # fov = 2 * atan((sensor_size/2) / focal_len)
        "x_axis_rads": 1.8260405563703   # fov = 2 * atan((sensor_size/2) / focal_len)
    },
}


# These are the 3D bounding boxes of each SKU, at zero rotation/translation
BBOX_SKU_1 = np.array(
    [[-0.0303775,	-0.013597,	-0.014708],
     [0.0303775,	-0.013597,	-0.014708],
     [0.0303775,	-0.013597,	0.014708],
     [-0.0303775,	-0.013597,	0.014708],
     [-0.0303775,	0.013597,	-0.014708],
     [0.0303775,	0.013597,	-0.014708],
     [0.0303775,	0.013597,	0.014708],
     [-0.0303775,	0.013597,	0.014708]]
)

BBOX_SKU_2 = np.array(
    [[-0.027383,	-0.022344,	-0.020727],
     [0.027383,	    -0.022344,	-0.020727],
     [0.027383,	    -0.022344,	0.020727],
     [-0.027383,	-0.022344,	0.020727],
     [-0.027383,	0.022344,	-0.020727],
     [0.027383,	    0.022344,	-0.020727],
     [0.027383,	    0.022344,	0.020727],
     [-0.027383,	0.022344,	0.020727]]
)

BBOX_SKU_3 = np.array(
    [[-0.026696,	-0.0165905,	-0.0215945],
     [0.026696,	    -0.0165905,	-0.0215945],
     [0.026696,	    -0.0165905,	0.0215945],
     [-0.026696,	-0.0165905,	0.0215945],
     [-0.026696,	0.0165905,	-0.0215945],
     [0.026696,	    0.0165905,	-0.0215945],
     [0.026696,	    0.0165905,	0.0215945],
     [-0.026696,	0.0165905,	0.0215945]]
)


def get_files_in_dir(directory: Path, file_extension: str) -> (List, int):
    """Get list of files in a directory whose names end with a given pattern. Eg. .rgb.png"""
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    files_list = sorted(directory.glob('*' + file_extension))
    num_files = len(files_list)
    if num_files == 0:
        raise ValueError(f"No files found matching '{file_extension}' in dir '{directory}'")

    return files_list, num_files


def get_renderid(fname: Path) -> str:
    """Get the renderid of an image. The render id is the numerical index.
    Eg: 12.rgb.png -> renderid = 12
    """
    return fname.name.split('.')[0]


def construct_camera_intrinsics(res_x: int, res_y: int, focal_len_mm: float, fov_x: float, fov_y: float):
    """Construc the camera's intrinsics matrix.
        It is defined by:
            [[fx, 0, cx],
             [0, fy, cy],
             [0, 0, 1  ]].
        Where,
            fx,fy = focal length in pixels alongs x,y axis
            cx, cy = image center (Principal point of image)

    Args:
        res_x: Image width in pixels
        res_y: Image height in pixels
        focal_len_mm: Focal length of camera in mm
        fov_x: Horizontal Field of view, in radians
        fov_y: Vertical Field of view, in radians

    Returns:
        np.ndarray: camera_intrinsics_matrix. Shape: [3, 3]
    """
    intr = np.eye(3, dtype=np.float32)

    fx = (res_x / 2) / math.tan(fov_x / 2)
    fy = (res_y / 2) / math.tan(fov_y / 2)
    cx = res_x / 2
    cy = res_y / 2

    intr[0, 0] = fx
    intr[1, 1] = fy
    intr[0, 2] = cx
    intr[1, 2] = cy

    return intr


def get_sku_bbox_corners(sku_id: int):
    """Get the 8 corners of the 3D bounding box encompassing a given SKU at default pose.
    There are 3 SKU IDs and their 3D BBOX values at default pose are hard-coded"""
    sku_bbox = {
        1: BBOX_SKU_1,
        2: BBOX_SKU_2,
        3: BBOX_SKU_3
    }

    return sku_bbox[sku_id]


def calculate_3d_bboxes_in_image(f_rgb: Path, f_info: Path, f_segments: Path, cam_intr: np.ndarray, sku_ids: Dict):
    """Calc the 3D bboxes of all objs in an img

    Args:
        f_rgb: Filename of rgb image
        f_info: Filename of info.json. It contains the position and orientation of all the objects in scene.
        f_segments: Filename of the segments image (contains mask of each object).
        cam_intr: Camera Intrinsics Matrix
        sku_ids: Mapping between render ids and sku id. All objects in each render are the same SKU.
    """
    render_id_rgb = get_renderid(f_rgb)
    render_id_info = get_renderid(f_info)
    if render_id_rgb != render_id_info:
        raise ValueError(f"The RGB file ({f_rgb.name}) and Info file ({f_info.name}) do not match."
                         f" They are of different render IDs.")

    with f_info.open() as fd:
        info = json.load(fd)

    # Read the camera extrinsics from the info.json
    cam_extr = np.array(info["camera_transform"]).reshape((4, 4)).T
    print('\ncam_extr:\n', cam_extr)
    cam_rot = cam_extr[:3, :3]
    print('cam rotation (xyz, degrees): ', R.from_matrix(cam_rot).as_euler('xyz', degrees=True))
    print(f'cam translation (xyz, meters): ', cam_extr[:3, 3].T)

    # Get list of all objects and their pose from info.json
    list_objs = info["objects"]

    # Get the SKU ID and it's default 3D bbox
    sku_id = sku_ids[render_id_rgb]
    sku_bbox = get_sku_bbox_corners(sku_id)  # Shape: (8, 3) -> 8 corners of bbox
    print('\nsku_bbox default (no rot or translation): \n', sku_bbox)

    # Get the rotated bbox for each obj in image
    obj_bboxes = dict()
    world_aligned_bbox = dict()  # This min/max corners of world-aligned bbox of each object is present in json
    list_objs.reverse()  # Invert list so that last object (ones on top) come first
    for obj_info in list_objs:
        # Construct 4x4 transformation matrix of each object from it's position and orientation
        obj_translation = obj_info["position"]
        obj_orientation = obj_info["orientation"]  # Orientation in Quaternion XYZW
        obj_rot_mat = R.from_quat(obj_orientation).as_matrix()
        obj_tranform_mat = np.eye(4, dtype=np.float32)
        obj_tranform_mat[:3, :3] = obj_rot_mat
        obj_tranform_mat[:3, 3] = np.array(obj_translation).T

        # Rotate default bbox
        obj_bbox_default = sku_bbox.copy().T  # Shape: [3, 8]
        obj_bbox_default = np.concatenate((obj_bbox_default, np.ones((1, 8))), axis=0)  # Shape: [4, 8], add homogenous coords
        obj_bbox_world = obj_tranform_mat @ obj_bbox_default  # Shape: [4, 8]  # Apply transformation

        # Collect in list
        obj_id = obj_info["id"]
        obj_bboxes[obj_id] = obj_bbox_world

        world_aligned_bbox[obj_id] = {
            "bbox_min": obj_info["bbox_min"],
            "bbox_max": obj_info["bbox_max"],
        }

        # TEST: SELECT PARTICULAR OBJ
        if TEST_OBJ_ID is not None and obj_id == TEST_OBJ_ID:
            print(f'Obj-{obj_id} Obj Transform Matrix:\n', obj_tranform_mat)
            # print(f'Obj-{obj_id} Bbox World Coords:\n', obj_bbox_world.T)

    # Project each 3D bbox to the RGB image
    rgb = cv2.imread(str(f_rgb), cv2.IMREAD_COLOR)
    segments = cv2.imread(str(f_segments), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    # for bbox_3d in list_obj_bboxes:
    for obj_id, bbox_3d in obj_bboxes.items():
        # TEST: SELECT PARTICULAR OBJ
        if TEST_OBJ_ID is not None and obj_id != TEST_OBJ_ID:
            continue

        # Skip if object is not visible
        obj_mask = (segments == obj_id).astype(np.uint8)  # Get mask of obj
        obj_mask = cv2.erode(obj_mask, np.ones((3, 3), np.uint8), iterations=1)  # Remove noise in mask
        if np.count_nonzero(obj_mask) == 0:
            # If mask is empty, skip this object
            continue
        else:
            obj_mask = obj_mask.astype(np.bool)

        # Get the world axis-aligned bbox (to compare and see we're getting sensible values)
        print(f'Obj-{obj_id} Bbox world coords (after applying transformation to sku bbox):\n', bbox_3d.T)
        w_aligned_bbox_min = world_aligned_bbox[obj_id]["bbox_min"]
        w_aligned_bbox_max = world_aligned_bbox[obj_id]["bbox_max"]
        print(f'Obj-{obj_id} World Aligned Bbox:\n  Min: {w_aligned_bbox_min}\n  Max: {w_aligned_bbox_max}')

        # Convert from world coords to camera coords
        bbox3d_cam_coords = cam_extr @ bbox_3d  # Shape: [4, 8]
        bbox3d_cam_coords = bbox3d_cam_coords[:3, :]  # Shape: [3, 8]
        print(f'Obj-{obj_id} Bbox cam coords (X-right, Y-Up):\n', bbox3d_cam_coords.T)

        # Rotation to convert (Y-up, X-right) camera to (Y-down, X-right) camera system for direct projection to pixels
        rot_x_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        bbox3d_cam_coords = rot_x_180 @ bbox3d_cam_coords
        print(f'Obj-{obj_id} Bbox rotated cam coords (X-right, Y-Down):\n', bbox3d_cam_coords.T)

        # Project to pixels
        bbox_px = cam_intr @ bbox3d_cam_coords
        bbox_px = bbox_px.T  # Shape: [8, 3]
        bbox_px = bbox_px / bbox_px[:, 2, np.newaxis]  # Normalize
        bbox_px = bbox_px[:, :2]
        print(f'Obj-{obj_id} Bbox pixel coords:\n', bbox_px)

        # Generate random color
        rand_col = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Draw points on image
        for pt_px in bbox_px:
            rgb = cv2.circle(rgb, tuple(pt_px.round().astype(np.int)), 2, rand_col, 1)

        # Colorize the mask
        rgb[obj_mask] = rand_col

        # Save output image
        fname = f_rgb.parent / (render_id_rgb + EXT_VIZ_BBOX)
        print(fname)
        cv2.imwrite(str(fname), rgb)
        print(f'\nSaved output image {fname}')

        if TEST_OBJ_ID is not None:
            # If testing enabled for single object, then quit after processing that obj
            print('Test mode - quitting')
            break


@hydra.main(config_path='.', config_name='config')
def main(cfg: DictConfig):
    # Seed random
    random.seed(SEED)

    dir_data = Path(cfg.dir_data)
    if not dir_data.is_dir():
        raise ValueError(f'Not a directory: {dir_data}')

    # Get all renders
    files_rgb, num_rgb = get_files_in_dir(dir_data, EXT_RGB)
    files_info, num_info = get_files_in_dir(dir_data, EXT_INFO)
    files_segments, num_segments = get_files_in_dir(dir_data, EXT_SEGMENTS)
    if num_rgb != num_info:
        raise ValueError(f"Unequal num of rgb ({num_rgb}) and info ({num_info}) files in dir '{dir_data}'")

    # Read Metadata -> Get SKU ID for each Render and camera information (resolution and focal len)
    f_metadata = dir_data / FNAME_METADATA
    # There are 3 unique SKUs (type of geometry). ALL OBJECTS IN EACH RENDER ARE THE SAME SKU
    sku_ids = dict()  # Key =  Render ID, Value = SKU ID
    with jsonlines.open(f_metadata) as reader:
        for idx, obj in enumerate(reader):
            if idx == 0:
                # Get camera info. It is the same for every render.
                CAM_DATA["resolution"] = obj["render"]["resolution"]
                CAM_DATA["focal_len_mm"] = obj["scene"]["camera"]["focal_length"]

            render_id = str(obj['render_id'])
            sku_id = obj['scene']['objects'][0]['id']
            sku_ids[render_id] = sku_id

    cam_intr = construct_camera_intrinsics(CAM_DATA["resolution"]["w"], CAM_DATA["resolution"]["h"],
                                           CAM_DATA["focal_len_mm"], CAM_DATA["field_of_view"]["x_axis_rads"],
                                           CAM_DATA["field_of_view"]["y_axis_rads"])

    print('cam_intr:\n', cam_intr)

    calculate_3d_bboxes_in_image(files_rgb[0], files_info[0], files_segments[0], cam_intr, sku_ids)




if __name__ == '__main__':
    main()
