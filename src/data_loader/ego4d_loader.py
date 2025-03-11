import os


from typing import Tuple

import cv2
import torch
import numpy as np
from src.data_loader.utils import get_joints_from_mano_mesh
from src.utils import read_json, save_json
from torch.utils.data import Dataset
from tqdm import tqdm
from src.data_loader.joints import Joints
from src.constants import MANO_MAT
import pandas as pd

import json
from src.data_loader.utils import crop_and_resize

SCALE = 1.3
CROP_SIZE = 224

class EGO4D_DB(Dataset):
    """Class to load samples from the Ego4D dataset.
    Inherits from the Dataset class in  torch.utils.data.
    Can be used for the supervised learning.
    Camera matrix is unity to fit with the sample augmenter.
    """

    def __init__(self, root_dir: str, split: str = "train", datasets_scale: str = "1m"):
        self.root_dir = root_dir
        self.split = split
        
        self.datasets_scale = datasets_scale

        self.anno_list, self.img_list = self.get_joints_labels_and_images()
        
        self.id_to_index_dict = self.get_initialize_id_to_index()
        
        self.img_dict = {item["id"]: item for item in self.img_list}
        self.joints = Joints()
        
    def get_joints_labels_and_images(self) -> Tuple[dict, dict]:
        """Returns the dictionary conatinign the bound box of the image and dictionary
        containig image information.

        Returns:
            Tuple[dict, dict]: joints, image_dict
                image_dict
                    - `name` - Image name in the form
                        of `youtube/VIDEO_ID/video/frames/FRAME_ID.png`.
                    - `width` - Width of the image.
                    - `height` - Height of the image.
                    - `id` - Image ID.
                joints
                    - `joints` - 21 joints, containing bound box limits as vertices.
                    - `is_left` - Binary value indicating a right/left hand side.
                    - `image_id` - ID to the corresponding entry in `images`.
                    - `id` - Annotation ID (an image can contain multiple hands).
        """
        
        data_json_path = os.path.join(self.root_dir, f"annotations/Ego4D/Hand100M_Ego4D_{self.datasets_scale}_v1-1.json")
        
        data_json = read_json(data_json_path)
        print(f"\nEgo4D {self.datasets_scale} JSON file loaded successfully.")
    
        images_dict = data_json["images"]
        annotations_dict = data_json["annotations"]

        print(f"A total of {len(images_dict)} images were read.")
        print(f"A total of {len(annotations_dict)} data items were read.")
        return annotations_dict, images_dict

    def get_initialize_id_to_index(self) -> dict:
        """Returns the dictionary conatinign the (key: hand_id, value: db_idx)
        """
        data_json_path = os.path.join(self.root_dir, f"annotations/Ego4D/Hand100M_Ego4D_{self.datasets_scale}_v1-1.json")
        data_json = read_json(data_json_path)
  
        annotations_dict = data_json["annotations"]
        
        id_to_index_dict = {annotation["hand_id"]: index for index, annotation in enumerate(annotations_dict)}
        
        assert len(id_to_index_dict) == len(annotations_dict)
        
        return id_to_index_dict
 
    def __len__(self):
        return len(self.anno_list)
        
    def __getitem__(self, idx: int) -> dict:
        """Returns a sample corresponding to the index.

        Args:
            idx (int): index

        Returns:
            dict: item with following elements.
                "image" in opencv bgr format.
                "K": camera params
                "joints3D": 3D coordinates of joints in AIT format.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(
            self.root_dir, self.img_dict[self.anno_list[idx]["image_id"]]["file_name"]
        )
        
        img = cv2.cvtColor(
            cv2.imread(img_name), cv2.COLOR_BGR2RGB
        )
        
        boxes = json.loads(self.anno_list[idx]["boxes"])
        img_crop = crop_and_resize(img, boxes, SCALE, CROP_SIZE)

        joints25D = torch.tensor(self.anno_list[idx]["keypoint_25d"])
        joints25D = joints25D.view(21,3)
        
        # Thougth the joints_raw to keep the original joints value:
        joints_raw = joints25D.clone()
        
        # Get the image coordinate from the json file
        joints25D[:, 0] *= img_crop.shape[1]
        joints25D[:, 1] *= img_crop.shape[0]
        
        if self.anno_list[idx]["left_right"] == 'Left':
            # flipping horizontally to make it right hand
            img_crop = cv2.flip(img_crop, 1)
            # width - x coord
            # Change the coordinate:
            joints25D[:, 0] = img_crop.shape[1] - joints25D[:, 0]
            joints_raw[:, 0] = 1 - joints_raw[:, 0]

        # because image is cropped and rotated with the 2d projections of these coordinates.
        # It needs to have depth as 1.0 to not cause problems. For procrustes use "joints_raw"
        joints25D[..., -1] = 1.0
        camera_param = torch.eye(3).float()
        joints_valid = torch.zeros_like(joints25D[..., -1:])
        
        positive_sample = str(self.anno_list[idx]["positive_sample"][0])
        positive_sample_idx = self.id_to_index_dict[positive_sample]
        
        distance = self.anno_list[idx]["distance"][0]
        
        hand_id = int(self.anno_list[idx]["hand_id"])

        sample = {
            "image": img_crop,
            "image_name": img_name,
            "hand_id": hand_id,
            "K": camera_param,
            "joints3D": joints25D,
            "joints_valid": joints_valid,
            "joints_raw": joints_raw,
            "positive_sample": positive_sample,
            "positive_sample_idx": positive_sample_idx,
            "distance": distance,
        }

        return sample