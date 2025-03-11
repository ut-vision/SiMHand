import os
from typing import Tuple

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd

import json
from utils.handutils import read_json, crop_and_resize
import utils.handutils as handutils

SCALE = 1.5
CROP_SIZE = 224

class Ego4DHandsDataset(Dataset):
    """Class to load samples from the EGO4D dataset.
    Inherits from the Dataset class in  torch.utils.data.
    Note: The keypoints are mapped to format used at AIT.
    Refer to joint_mapping.json in src/data_loader/utils.
    """

    def __init__(
        self, transform, root_dir: str, data_split: str = "train", datasets_scale: str = "100k", logger = None
    ):
        self.transform = transform
        self.root_dir = root_dir
        self.data_split = data_split
        self.datasets_scale = datasets_scale
        self.logger = logger
        self.anno_list, self.img_list = self.get_joints_labels_and_images()
        self.img_dict = {item["id"]: item for item in self.img_list}
        self.crop = False # ego4d_hands datasets is already crop datasets.

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

        joints25D = torch.tensor(self.anno_list[idx]["keypoint_25d"]).float()
        joints25D = joints25D.view(21,3)
        # Thougth the joints_raw to keep the original joints value:
        joints_raw = joints25D.clone()
        
        # Transfer the joints25D to the image coordinate:
        joints25D[:, 0] *= CROP_SIZE
        joints25D[:, 1] *= CROP_SIZE
        joints_img = joints25D[:, :2].clone().numpy()

        if self.anno_list[idx]["left_right"] == 'Left':
            # flipping horizontally to make it right hand
            img_crop = cv2.flip(img_crop, 1)
            # width - x coord
            # Change the coordinate:
            joints25D[:, 0] = img_crop.shape[1] - joints25D[:, 0]
            joints_raw[:, 0] = 1 - joints_raw[:, 0]
        
        # joints3D = torch.tensor(self.bbox[idx_]["joints"]).float()
        
        # TODO: transform to pil image, but not support float, cv2.imread==pil.read?
        img_crop = self.transform(img_crop.astype(np.float32))

        # because image is cropped and rotated with the 2d projections of these coordinates.
        # It needs to have depth as 1.0 to not cause problems. For procrustes use "joints_raw"
        joints25D[..., -1] = 1.0
        camera_param = torch.eye(3).float()
        joints_valid = torch.ones_like(joints25D[..., -1:])
            
        data = {
            "image": img_crop,
            "K": camera_param,
            "joints2D": joints_img,
            "joints3D": joints25D,
            "joints_valid": joints_valid,
            "image_name": img_name,
        }
        return data

    def get_sample(self, index):
        data = self.__getitem__(index)
        clr = data["image"]
        intr = data["K"]
        # 1
        # 2 process and get 2d kp in img space
        # due to the friehands is all right hand, so don't need to handle the left-right hand problem
        kp2d = data["joints2D"]
        # 3 process and get 3d kp in camera space
        # kp3d = data["joints3D"]
        center = handutils.get_annot_center(kp2d)
        # 5 tranform the index of the kp, making it identical to other datasets.

        # kp3d = kp3d
        # In fh datasets don't need to set the vis param
        # vis = np.ones(20)
        # kp3d = kp3d / 1000  # transform mm to m（compatible with other datasets）
        # 5
        # 3 calculate the parameters for cropping from 2dkp
        if self.crop:
            center = np.asarray([int(clr.size[0] / 2), int(clr.size[1] / 2)])
            my_scale = clr.size[0]  # Here the scale is defined as the pixel range of cropping
        else:
            center = handutils.get_annot_center(kp2d)
            my_scale = handutils.get_ori_crop_scale(mask=None, side=None, mask_flag=False, kp2d=kp2d)

        sample = {
            'index': index,
            'clr': clr,
            'kp2d': kp2d,
            'center': center,
            'my_scale': my_scale,
            # ego4d_hands don't have the 3D joints
            # 'joint': kp3d,
            'intr': intr,
            # 'vis': vis,
        }
        
        return sample

if __name__ == "__main__":
    from.ego4d_hands import Ego4DHandsDataset
    from tqdm import tqdm
    import torchvision.transforms as transforms

    data_root = 'PATH/TO/EGO4D/HANDS/DATASET'
    data_split = 'train'
    dataset = Ego4DHandsDataset(
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ToPILImage(),
                    ]),
                root_dir=os.path.join(data_root, "Hand100M"),
                data_split=data_split,
                # @new for ego4d_hands
                datasets_scale = "100k",
                # train_ratio=self.config.train_ratio,
                # train_ratio=0.9999999999,
                # subset_ratio=subset_ratio,
                # logger = logger,
            )
    for i in tqdm(range(len(dataset))):
        print(f"The number of data we get is No. {i}")
        sample = dataset[i]
        print(sample)