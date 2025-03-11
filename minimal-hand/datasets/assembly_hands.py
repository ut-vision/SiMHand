# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.utils.data
import cv2
import os
import os.path as osp
import random
from copy import deepcopy
from .assemblyhands.utils.preprocessing import (
    load_img,
    load_crop_img,
    update_params_after_crop,
    load_skeleton,
    get_bbox,
    process_bbox,
    augmentation,
    transform_input_to_output_space,
    trans_point2d,
)
from .assemblyhands.utils.transforms import cam2pixel, pixel2cam, Camera
from .assemblyhands.utils.transforms import world2cam_assemblyhands as world2cam
from .assemblyhands.utils.transforms import cam2world_assemblyhands as cam2world
from .assemblyhands.utils.vis import vis_keypoints, vis_3d_keypoints
from copy import deepcopy
import json
from pycocotools.coco import COCO
import utils.handutils as handutils
from PIL import Image

ANNOT_VERSION = "v1-1"
# IS_DEBUG = True
IS_DEBUG = False
N_DEBUG_SAMPLES = 200

import torch
import os
import os.path as osp
from tqdm import tqdm


class AssemblyHandsDataset(torch.utils.data.Dataset):
    def __init__(self, transform, cfg, data_root="data/assemblyhands", data_split='train', hand_side='right',
                 modality='ego', njoints=21, use_cache=True, visual=False, ah_crop=False, subset_ratio = None, logger = None):
        if not os.path.exists(data_root):
            raise ValueError("data_root: %s not exist" % data_root)

        self.name = 'ah'

        # TODO: now dataset has no test part, change it to val
        # assert data_split != 'test'
        self.mode = data_split
        # if self.mode == 'test':
        #     print("Warn: Dataset has no testset, change to val")
        #     self.mode = 'val'
        self.img_path = osp.join(data_root, "images/")
        self.annot_path = osp.join(data_root, "annotations/")
        self.modality = modality
        self.transform = transform
        self.joint_num = njoints
        self.reslu = cfg.input_img_shape
        self.cfg = cfg
        self.visual = visual
        self.hand_side = hand_side
        self.crop = ah_crop  # whether to use AssemblyHands crop, or handataset crop
        self.subset_ratio = subset_ratio
        
        self.root_joint_idx = {"right": 20, "left": 41}  # TODO No
        self.joint_type = {
            "right": np.arange(0, self.joint_num),
            "left": np.arange(self.joint_num, self.joint_num * 2),
        }
        self.skeleton = load_skeleton(
            osp.join(self.annot_path, "skeleton.txt"), self.joint_num * 2
        )

        self.datalist = []
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = []
        n_skip = 0
        # load annotation
        self.logger = logger
        self.logger.info(self.mode)
        self.logger.info(f"Load annotation from  {self.annot_path}, mode: {self.mode}")
        # print(self.mode)
        # print(f"Load annotation from  {self.annot_path}, mode: {self.mode}")
        data_mode = self.mode
        if IS_DEBUG and self.mode.startswith("train"):
            self.logger.info(">>> DEBUG MODE: Loading val data during training")
            # print(">>> DEBUG MODE: Loading val data during training")
            data_mode = "val"
        self.invalid_data_file = os.path.join(
            self.annot_path, data_mode, f"invalid_{data_mode}_{self.modality}.txt"
        )
        db = COCO(
            osp.join(
                self.annot_path,
                data_mode,
                "assemblyhands_"
                + data_mode
                + f"_{self.modality}_data_{ANNOT_VERSION}.json",
            )
        )
        with open(
                osp.join(
                    self.annot_path,
                    data_mode,
                    "assemblyhands_"
                    + data_mode
                    + f"_{self.modality}_calib_{ANNOT_VERSION}.json",
                )
        ) as f:
            cameras = json.load(f)["calibration"]
        with open(
                osp.join(
                    self.annot_path,
                    data_mode,
                    "assemblyhands_" + data_mode + f"_joint_3d_{ANNOT_VERSION}.json",
                )
        ) as f:
            joints = json.load(f)["annotations"]

        self.logger.info("Get bbox and root depth from groundtruth annotation")
        # print("Get bbox and root depth from groundtruth annotation")
        invalid_data_list = None
        if osp.exists(self.invalid_data_file):
            with open(self.invalid_data_file) as f:
                lines = f.readlines()
            if len(lines) > 0:
                invalid_data_list = [line.strip() for line in lines]
        else:
            self.logger.info("Invalid data file does not exist. Checking the validity of generated crops")
            # print("Invalid data file does not exist. Checking the validity of generated crops")
            f = open(self.invalid_data_file, "w")

        annot_list = db.anns.keys()
        for i, aid in enumerate(tqdm(annot_list)):
            ann = db.anns[aid]
            image_id = ann["image_id"]
            img = db.loadImgs(image_id)[0]

            seq_name = str(img["seq_name"])
            camera_name = img["camera"]
            frame_idx = img["frame_idx"]
            file_name = img["file_name"]
            img_path = osp.join(self.img_path, file_name)
            assert osp.exists(img_path), f"Image path {img_path} does not exist"

            K = np.array(
                cameras[seq_name]["intrinsics"][camera_name + "_mono10bit"],
                dtype=np.float32,
            )
            Rt = np.array(
                cameras[seq_name]["extrinsics"][f"{frame_idx:06d}"][
                    camera_name + "_mono10bit"
                    ],
                dtype=np.float32,
            )
            retval_camera = Camera(K, Rt, dist=None, name=camera_name)
            campos, camrot, focal, princpt = retval_camera.get_params()

            joint_world = np.array(
                joints[seq_name][f"{frame_idx:06d}"]["world_coord"], dtype=np.float32
            )
            joint_cam = world2cam(joint_world, camrot, campos)
            joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]

            joint_valid = np.array(ann["joint_valid"], dtype=np.float32).reshape(
                self.joint_num * 2
            )
            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            # joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
            # joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]

            abs_depth = {
                "right": joint_cam[self.root_joint_idx["right"], 2],
                "left": joint_cam[self.root_joint_idx["left"], 2],
            }
            cam_param = {"focal": focal, "princpt": princpt}
            for hand_id, hand_type in enumerate(["right", "left"]):
                if ann["bbox"][hand_type] is None:
                    continue
                hand_type_valid = np.ones(1, dtype=np.float32)

                img_width, img_height = img["width"], img["height"]
                bbox = np.array(ann["bbox"][hand_type], dtype=np.float32)  # x,y,x,y
                x0, y0, x1, y1 = bbox
                original_bbox = [x0, y0, x1 - x0, y1 - y0]  # x,y,w,h
                bbox = process_bbox(
                    original_bbox, (img_height, img_width), scale=1.75
                )  # bbox = original_bbox

                joint_valid_single_hand = deepcopy(joint_valid)
                inv_hand_id = abs(1 - hand_id)
                # make invlid for the other hand
                joint_valid_single_hand[
                inv_hand_id * self.joint_num: (inv_hand_id + 1) * self.joint_num
                ] = 0
                if invalid_data_list is not None:
                    crop_name = f"{file_name},{hand_id}"
                    if crop_name in invalid_data_list:  # skip registred invalid samples
                        n_skip += 1
                        continue
                else:  # first run to check the validity of generated crops
                    if sum(joint_valid_single_hand) < 10:
                        n_skip += 1
                        f.write(f"{file_name},{hand_id}\n")
                        continue
                    try:
                        load_crop_img(
                            img_path,
                            bbox,
                            joint_img.copy(),
                            joint_world.copy(),
                            joint_valid_single_hand.copy(),
                            deepcopy(retval_camera),
                        )
                    except:
                        n_skip += 1
                        f.write(f"{file_name},{hand_id}\n")
                        continue

                joint = {
                    "cam_coord": joint_cam,
                    "img_coord": joint_img,
                    "world_coord": joint_world,
                    "valid": joint_valid_single_hand,
                }  # joint_valid
                data = {
                    "img_path": img_path,
                    "seq_name": seq_name,
                    "cam_param": cam_param,
                    "bbox": bbox,
                    "original_bbox": original_bbox,
                    "joint": joint,
                    "hand_type": hand_type,
                    "hand_type_valid": hand_type_valid,
                    "abs_depth": abs_depth,
                    "file_name": img["file_name"],
                    "seq_name": seq_name,
                    "cam": camera_name,
                    "frame": frame_idx,
                    "retval_camera": retval_camera,
                }
                if hand_type == "right" or hand_type == "left":
                    self.datalist_sh.append(data)
                else:
                    self.datalist_ih.append(data)
                if seq_name not in self.sequence_names:
                    self.sequence_names.append(seq_name)

            if IS_DEBUG and i >= N_DEBUG_SAMPLES - 1:
                self.logger.info(">>> DEBUG MODE: Loaded %d samples" % N_DEBUG_SAMPLES)
                # print(">>> DEBUG MODE: Loaded %d samples" % N_DEBUG_SAMPLES)
                break
        self.datalist = self.datalist_sh + self.datalist_ih
        assert len(self.datalist) > 0, "No data found."
        if not osp.exists(self.invalid_data_file):
            f.close()
        self.logger.info(
            "Number of annotations in single hand sequences: "
            + str(len(self.datalist_sh))
        )
        # print(
        #     "Number of annotations in single hand sequences: "
        #     + str(len(self.datalist_sh))
        # )
        self.logger.info(
            "Number of annotations in interacting hand sequences: "
            + str(len(self.datalist_ih))
        )
        # print(
        #     "Number of annotations in interacting hand sequences: "
        #     + str(len(self.datalist_ih))
        # )
        self.logger.info("Number of skipped annotations: " + str(n_skip))
        # print("Number of skipped annotations: " + str(n_skip))

        self.datalist = self.datalist[:int(len(self.datalist) * self.subset_ratio)]
        if self.subset_ratio != 1.0:
            self.logger.info(f"The {self.mode}ing dataset has been sampled, and only the frist {self.subset_ratio * 100}% of the data will be used for training")
            # print(f"The training dataset has been sampled, and only the frist {self.subset_ratio * 100}% of the data will be used for training")
        

    def handtype_str2array(self, hand_type):
        if hand_type == "right":
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == "left":
            return np.array([0, 1], dtype=np.float32)
        elif hand_type == "interacting":
            return np.array([1, 1], dtype=np.float32)
        else:
            assert 0, print("Not supported hand type: " + hand_type)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, bbox, joint, hand_type, hand_type_valid = (
            data["img_path"],
            data["bbox"],
            data["joint"],
            data["hand_type"],
            data["hand_type_valid"],
        )
        joint_world = joint["world_coord"]
        joint_img = joint["img_coord"].copy()
        joint_valid = joint["valid"].copy()
        hand_type = self.handtype_str2array(hand_type)

        if not self.crop:
            joint_cam = joint["cam_coord"].copy()
            img = load_img(img_path, bbox)
            img = img / 255.0
            retval_camera = data["retval_camera"]
        else:
            img, bbox, joint_img, joint_cam, joint_valid, retval_camera = load_crop_img(
                img_path,
                bbox,
                joint_img,
                joint_world,
                joint_valid,
                deepcopy(data["retval_camera"]),
            )
            img = img / 255.0
        joint_coord = np.concatenate((joint_img, joint_cam[:, 2, None]), 1)  # joint定义
        # joint_coord = joint_cam
        # augmentation
        if self.crop:
            img, joint_coord, joint_valid, hand_type, inv_trans = augmentation(
                img,
                bbox,
                joint_coord,
                joint_valid,
                hand_type,
                self.mode,
                self.joint_type,
                no_aug=True,
            )
        rel_root_depth = np.array(
            [
                joint_coord[self.root_joint_idx["left"], 2]
                - joint_coord[self.root_joint_idx["right"], 2]
            ],
            dtype=np.float32,
        ).reshape(1)
        root_valid = (
            np.array(
                [
                    joint_valid[self.root_joint_idx["right"]]
                    * joint_valid[self.root_joint_idx["left"]]
                ],
                dtype=np.float32,
            ).reshape(1)
            if hand_type[0] * hand_type[1] == 1
            else np.zeros((1), dtype=np.float32)
        )
        # transform to output heatmap space
        (
            joint_coord_hm,
            joint_valid,
            rel_root_depth,
            root_valid,
        ) = transform_input_to_output_space(
            joint_coord.copy(),
            joint_valid,
            rel_root_depth,
            root_valid,
            self.root_joint_idx,
            self.joint_type,
        )
        # TODO: transform to pil image, but not support float, cv2.imread==pil.read?
        img = self.transform(img.astype(np.float32))

        # update camera parameters after resize to cfg.input_img_shape
        retval_camera.update_after_resize((bbox[3], bbox[2]), self.cfg.input_img_shape)
        campos, camrot, focal, princpt = retval_camera.get_params()
        cam_param = {"focal": focal, "princpt": princpt}

        inputs = {"img": img, "idx": idx}
        targets = {
            "joint_coord": joint_coord_hm,
            "joint_cam": joint_cam,
            "_joint_coord": joint_coord,
            "rel_root_depth": rel_root_depth,
            "hand_type": hand_type,
        }
        meta_info = {
            "joint_valid": joint_valid,
            "root_valid": root_valid,
            "hand_type_valid": hand_type_valid,
            "seq_name": data["seq_name"],
            "cam": data["cam"],
            "frame": int(data["frame"]),
            "cam_param_updated": cam_param,
        }
        if self.crop:
            meta_info["inv_trans"] = inv_trans
        return inputs, targets, meta_info

    def get_sample(self, index):
        # if self.mode not in ['val', 'train']:
        #     print("No {} set in Assembly Hands dataset!".format(self.mode))
        #     return None
        # 1 get data from the original dataset by __getitem__
        inputs, targets, meta_info = self.__getitem__(index)
        gts = self.datalist
        data = gts[index]
        clr = inputs["img"]
        # 1 
        # 2 process and get 2d kp in img space
        joint_coord_img = targets["joint_coord"].copy()
        joint_coord_img[:, 0] = (
                joint_coord_img[:, 0]
                / self.cfg.output_hm_shape[2]
                * self.cfg.input_img_shape[1]
        )
        joint_coord_img[:, 1] = (
                joint_coord_img[:, 1]
                / self.cfg.output_hm_shape[1]
                * self.cfg.input_img_shape[0]
        )
        # restore depth to original camera space
        joint_coord_img[:, 2] = (
                                        joint_coord_img[:, 2] / self.cfg.output_hm_shape[0] * 2 - 1
                                ) * (self.cfg.bbox_3d_size / 2)
        joint_coord_img[self.joint_type["right"], 2] += data["abs_depth"][
            "right"
        ]
        joint_coord_img[self.joint_type["left"], 2] += data["abs_depth"][
            "left"
        ]
        # 2            
        # 4 handle the left-right hand problem
        # if there is left hand, read its kp
        if data["hand_type"] in ['right', 'interaction'] and self.hand_side == 'right':
            kp2d = joint_coord_img[:21, :2].copy()
            # kp3d = joint_coord_img[:21].copy()
            kp3d = targets['joint_cam'][:21].copy()
            vis = meta_info["joint_valid"][:21].copy()
        # if there is right hand, read its kp
        elif data["hand_type"] in ['left', 'interaction'] and self.hand_side == 'left':
            kp2d = joint_coord_img[21:, :2].copy()
            # kp3d = joint_coord_img[21:].copy()
            kp3d = targets['joint_cam'][21:].copy()
            vis = meta_info["joint_valid"][21:].copy()
        # if none, then flip
        else:  # flip
            # print("flip")
            if self.hand_side == "right":
                kp2d = joint_coord_img[21:, :2].copy()
                # kp3d = joint_coord_img[21:].copy()
                kp3d = targets['joint_cam'][21:].copy()
                vis = meta_info["joint_valid"][21:].copy()
            if self.hand_side == "left":
                kp2d = joint_coord_img[:21, :2].copy()
                # kp3d = joint_coord_img[21:].copy()
                kp3d = targets['joint_cam'][:21].copy()
                vis = meta_info["joint_valid"][:21].copy()
            clr = clr.transpose(Image.FLIP_LEFT_RIGHT)
            center = handutils.get_annot_center(kp2d)
            center[0] = clr.size[0] - center[0]  # clr.size[0] represents width of image
            kp2d[:, 0] = clr.size[0] - kp2d[:, 0]
            kp3d[:, 0] = -kp3d[:, 0]
        # 4
        # 5 tranform the index of the kp, making it identical to other datasets.
        kp3d = kp3d[[20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16]]
        kp2d = kp2d[[20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16]]
        kp3d = kp3d / 1000
        vis = vis[[20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16]]
        # kp3d = kp3d/1000 # transform mm to m（compatible with other datasets）
        # 5
        # 3 calculate the parameters for cropping from 2dkp
        if self.crop:
            center = np.asarray([int(clr.size[0] / 2), int(clr.size[1] / 2)])
            my_scale = clr.size[0]  # Here the scale is defined as the pixel range of cropping
        else:
            center = handutils.get_annot_center(kp2d)
            my_scale = handutils.get_ori_crop_scale(mask=None, side=None, mask_flag=False, kp2d=kp2d)
        # 3
        sample = {
            'index': index,
            'clr': clr,
            'kp2d': kp2d,
            'center': center,
            'my_scale': my_scale,
            'joint': kp3d,
            'vis': vis,
        }

        return sample


if __name__ == "__main__":
    from .assemblyhands.utils.config import Config
    from torchvision import transforms
    from tqdm import tqdm

    ass_config = Config("AssemblyHands-Ego")
    dataset = AssemblyHandsDataset(
        transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
        ]),
        ass_config,
        data_root=os.path.join('./Minimal-Hand-pytorch/data', 'assemblyhands'),
        data_split='val',
        hand_side='right',
        njoints=21,
        ah_crop=True,
    )
    for i in tqdm(range(len(dataset))):
        if dataset[i][2]['frame'] == 13581:
            print(i)
    # print(dataset[278])
