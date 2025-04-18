import os
from torch import Tensor
from typing import List, Tuple
import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from .freihands.utils.transforms import cam2pixel, pixel2cam, Camera
from .freihands.utils.transforms import world2cam_assemblyhands as world2cam
from .freihands.utils.transforms import cam2world_assemblyhands as cam2world
from .freihands.utils.utils import convert_2_5D_to_3D, get_params_from_camera_param
from .freihands.utils.joints import Joints
from .freihands.utils.utils import read_json
from torch.utils.data import Dataset
import utils.handutils as handutils
from PIL import Image

BOUND_BOX_SCALE = 0.33

class FreihandsDataset(Dataset):
    """Class to load samples from the Freihand dataset.
    Inherits from the Dataset class in  torch.utils.data.
    Note: The keypoints are mapped to format used at AIT.
    Refer to joint_mapping.json in src/data_loader/utils.
    """

    def __init__(
        self, transform, root_dir: str, data_split: str, seed: int = 5, train_ratio: float = 1.0, subset_ratio = None, logger = None
    ):
        """Initializes the freihand dataset class, relevant paths and the Joints
        class for remapping of freihand formatted joints to that of AIT.

        Args:
            root_dir (str): Path to the directory with image samples.
        """
        self.transform = transform
        self.root_dir = root_dir
        self.split = data_split
        self.seed = seed
        self.train_ratio = train_ratio
        self.logger = logger
        self.labels = self.get_labels()
        self.scale = self.get_scale()   # No use in here
        self.camera_param = self.get_camera_param()
        self.img_names, self.img_path = self.get_image_names()
        self.indices = self.create_train_val_split(subset_ratio)
        # To convert from freihand to AIT format.
        self.joints = Joints()
        self.crop = False # Freihands datasets is already crop datasets.
        
    
    def create_train_val_split(self, subset_ratio) -> np.array:
        """Creates split for train and val data in freihand

        Raises:
            NotImplementedError: In case the split doesn't match test, train or val.

        Returns:
            np.array: array of indices
        """
        num_unique_images = len(self.camera_param)
        train_indices, val_indices = train_test_split(
            np.arange(num_unique_images),
            train_size=self.train_ratio,
            random_state=self.seed,
        )
        # Only use the subset_ratio param into the training datasets ----------------------------------- #
        if self.split == "train":
            train_indices = np.sort(train_indices)
            train_indices = np.concatenate(
                (
                    train_indices,
                    train_indices + num_unique_images,
                    train_indices + num_unique_images * 2,
                    train_indices + num_unique_images * 3,
                ),
                axis=0,
            )

            if subset_ratio != 1.0:
                train_indices = train_indices[:int(len(train_indices) * subset_ratio)]
                self.logger.info(f"The training dataset has been sampled, and only the frist {subset_ratio * 100}% of the data will be used for training")
                # print(f"The training dataset has been sampled, and only the frist {self.subset_ratio * 100}% of the data will be used for training")
            
            return train_indices
        
        elif self.split == "val":
            val_indices = np.sort(val_indices)
            val_indices = np.concatenate(
                (
                    val_indices,
                    val_indices + num_unique_images,
                    val_indices + num_unique_images * 2,
                    val_indices + num_unique_images * 3,
                ),
                axis=0,
            )
            return val_indices
        elif self.split == "test":
            test_indices = np.arange(len(self.camera_param))
            
            if subset_ratio != 1.0:
                test_indices = test_indices[:int(len(test_indices) * subset_ratio)]
                self.logger.info(f"The testing dataset has been sampled, and only the frist {subset_ratio * 100}% of the data will be used for evaling")
                # print(f"The training dataset has been sampled, and only the frist {self.subset_ratio * 100}% of the data will be used for training")
            
            return test_indices
        
        else:
            raise NotImplementedError
        
        
    def get_image_names(self) -> Tuple[List[str], str]:
        """Gets the name of all the files in root_dir.
        Make sure there are only image in that directory as it reads all the file names.

        Returns:
            List[str]: List of image names.
            str: base path for image directory
        """
        if self.split in ["train", "val"]:
            img_path = os.path.join(self.root_dir, "training", "rgb")
        else:
            img_path = os.path.join(self.root_dir, "evaluation", "rgb")
        img_names = next(os.walk(img_path))[2]
        img_names.sort()
        return img_names, img_path
    
    
    def get_labels(self) -> list:
        """Extacts the labels(joints coordinates) from the label_json at labels_path
        Returns:
            list: List of all the the coordinates(32650).
        """
        if self.split in ["train", "val"]:
            labels_path = os.path.join(self.root_dir, "training_xyz.json")
            return read_json(labels_path)
        else:
            labels_path = os.path.join(self.root_dir + "_eval", "evaluation_xyz.json")
            print(labels_path)
            return read_json(labels_path)
    

    def get_scale(self) -> list:
        """Extacts the scale from freihand data."""
        if self.split in ["train", "val"]:
            labels_path = os.path.join(self.root_dir, "training_scale.json")
        else:
            labels_path = os.path.join(self.root_dir, "evaluation_scale.json")
        return read_json(labels_path)

    def get_camera_param(self) -> list:
        """Extacts the camera parameters from the camera_param_json at camera_param_path.
        Returns:
            list: List of camera paramters for all images(32650)
        """
        if self.split in ["train", "val"]:
            camera_param_path = os.path.join(self.root_dir, "training_K.json")
        else:
            camera_param_path = os.path.join(self.root_dir,  "evaluation_K.json")
        return read_json(camera_param_path)

    
    def __len__(self):
        return len(self.indices)

    
    def create_sudo_bound_box(self, scale) -> Tensor:
        max_bound = torch.tensor([224.0, 224.0])
        min_bound = torch.tensor([0.0, 0.0])
        c = (max_bound + min_bound) / 2.0
        s = ((max_bound - min_bound) / 2.0) * scale
        bound_box = torch.tensor(
            [[0, 0, 0]]
            + [[s[0], s[1], 1]] * 5
            + [[-s[0], s[1], 1]] * 5
            + [[s[0], -s[1], 1]] * 5
            + [[-s[0], -s[1], 1]] * 5
        ) + torch.tensor([c[0], c[1], 0])
        return bound_box.float()

    
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
        idx_ = self.indices[idx]
        img_name = os.path.join(self.img_path, self.img_names[idx_])
        img = cv2.cvtColor(cv2.imread(img_name),cv2.COLOR_BGR2RGB)
        
        # TODO: transform to pil image, but not support float, cv2.imread==pil.read?
        img = self.transform(img.astype(np.float32))
        

        if self.labels is not None:
            camera_param = torch.tensor(self.camera_param[idx_ % 32560]).float()
            joints3D = torch.tensor(self.labels[idx_ % 32560]).float()
        else:
            camera_param = torch.tensor(self.camera_param[idx_]).float()
            joints2d_orthogonal = self.create_sudo_bound_box(scale=BOUND_BOX_SCALE)
            joints3D = convert_2_5D_to_3D(
                joints2d_orthogonal, scale=1.0, K=camera_param.clone()
            )
            
        joints_valid = torch.ones_like(joints3D[..., -1:])

        # ----------------------- joint_world -------------------------- #
        # In FreiHands only support the camera 3D joints keypoints
        _, _, focal, princpt = get_params_from_camera_param(camera_param)
        joint_cam = joints3D
        joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]
        
        data = {
            "image": img,
            "K": camera_param,
            "joints2D": joint_img,
            "joints3D": joint_cam,
            "joints_valid": joints_valid,
            # 方便检查错误：
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
        kp3d = data["joints3D"]
        center = handutils.get_annot_center(kp2d)
        # 5 tranform the index of the kp, making it identical to other datasets.

        kp3d = kp3d
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
            'joint': kp3d,
            'intr': intr,
            # 'vis': vis,
        }
        
        return sample


if __name__ == "__main__":
    from.freihand import FreihandsDataset
    from tqdm import tqdm

    data_root = 'PATH/TO/FREIHAND/DATASET'
    data_split = 'train'
    dataset = FreihandsDataset(
        root_dir=os.path.join(data_root, "FreiHAND/FreiHAND_pub_v2"),
        data_split=data_split,
        # train_ratio=self.config.train_ratio,
        train_ratio=1.0,
        )
    for i in tqdm(range(len(dataset))):
        print(f"The number of data we get is No. {i}")
        sample = dataset[i]
        print(sample)