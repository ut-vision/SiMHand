import numpy as np
import torch
import torch.utils.data
import cv2
import os
import os.path as osp
import sys
import yaml
import random
import pickle
from copy import deepcopy
import json
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets import DEFAULT_CACHE_DIR
CACHE_HOME = os.path.expanduser(DEFAULT_CACHE_DIR)
# @2024.05.10
# It looks like the handutils used in dexycb are not quite the same as the original version, 
# so I'll save both.
import utils.handutils_dy as handutils
from utils import align
from datasets import N_VALID_KEYPOINTS, VALID_SEQ_SAMPLE_RATE, REF_BONE_LINK, CENTER_ID
from datasets.utils import vis_kpt_3d

from utils.transforms import world2pixel, cam2pixel, pixel2cam, world2cam, cam2world

_SUBJECTS = [
    '20200709-subject-01',
    '20200813-subject-02',
    '20200820-subject-03',
    '20200903-subject-04',
    '20200908-subject-05',
    '20200918-subject-06',
    '20200928-subject-07',
    '20201002-subject-08',
    '20201015-subject-09',
    '20201022-subject-10',
]

_SERIALS = [
    '836212060125',
    '839512060362',
    '840412060917',
    '841412060263',
    '932122060857',
    '932122060861',
    '932122061900',
    '932122062010',
]

SUB2ID = {s: i for i, s in enumerate(_SUBJECTS)}
ID2SUB = {i: s for i, s in enumerate(_SUBJECTS)}
CAM2ID = {c: i for i, c in enumerate(_SERIALS)}
ID2CAM = {i: c for i, c in enumerate(_SERIALS)}
HAND2ID = {"right": 0, "left": 1} #, "interacting": 2}
ID2HAND = {v: k for k, v in HAND2ID.items()}

# SCALE_NORM = "per_frame" 
SCALE_NORM = "per_seq"


# Load intrinsics.
def intr_to_K(x):
    return torch.tensor(
        [[x['fx'], 0.0, x['ppx']], [0.0, x['fy'], x['ppy']], [0.0, 0.0, 1.0]],
        dtype=torch.float32)
    
def get_data_split(split, setup='s1'):
    if setup == 's1': # Unseen subjects.
        if split == 'train':
            subject_ind = [0, 1, 2, 3, 4, 5, 9]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        elif split == 'val':
            subject_ind = [6]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        elif split == 'test':
            subject_ind = [7, 8]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = list(range(100))
    elif setup == 's2': # Unseen camera views.
        if split == 'train':
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [0, 1, 2, 3, 4, 5]
        elif split == 'val':
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [6]            
        elif split == 'test':
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [7]
        sequence_ind = list(range(100))        
    elif setup == 's12-src':
        """
        train/val/test: sub 0-5,9/6/7,8
        src: cam 0-5
        trg1: cam6
        trg2: cam7
        """
        if split == 'train':
            subject_ind = [0, 1, 2, 3, 4, 5, 9]
            serial_ind = [0, 1, 2, 3, 4, 5]
        elif split == 'val':
            subject_ind = [6]
            serial_ind = [0, 1, 2, 3, 4, 5]
        elif split == 'test':
            subject_ind = [7, 8]
            serial_ind = [0, 1, 2, 3, 4, 5]
        sequence_ind = list(range(100))
    elif setup == 's12-trg1': # cam6
        if split == 'train':
            subject_ind = [0, 1, 2, 3, 4, 5, 9]
            serial_ind = [6]
        elif split == 'val':
            subject_ind = [6]
            serial_ind = [6]
        elif split == 'test':
            subject_ind = [7, 8]
            serial_ind = [6]
        sequence_ind = list(range(100))
    elif setup == 's12-trg2': # cam6
        if split == 'train':
            subject_ind = [0, 1, 2, 3, 4, 5, 9]
            serial_ind = [7]
        elif split == 'val':
            subject_ind = [6]
            serial_ind = [7]
        elif split == 'test':
            subject_ind = [7, 8]
            serial_ind = [7]
        sequence_ind = list(range(100))
    return subject_ind, serial_ind, sequence_ind

class DexYCBDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_root, data_split, hand_side="right", njoints=21, use_cache=True, 
                seq_load=False, seqlen=16, stride=16, setup='s1', unlabeled=False,
                no_img_load=False, pose_cache_path=None, vis=False, is_debug=False):
        self.name = 'dy'
        self.data_root = data_root
        self.data_split = data_split
        self.hand_side = hand_side
        self.modality = "rgb"
        self.clr_paths = []
        self.pts_2ds = []
        self.joints = []
        self.centers = []
        self.crop_scales = []
        self.njoints = njoints
        self.reslu = [256, 256]
        self.seq_load = seq_load
        self.seqlen = seqlen
        self.stride = stride
        self.setup = setup
        self.no_img_load = no_img_load
        self.unlabeled = unlabeled
        if pose_cache_path is not None:
            print("set pose cache", pose_cache_path)
            self.set_pose_cache(pose_cache_path)
        self.vis = vis
        self.is_debug = is_debug
        self.ref_bone_link = REF_BONE_LINK
        calib_file = os.path.join(self.data_root, f"annotations/calib_all.json")
        with open(calib_file) as f:
            self.calib = json.load(f)['calibration']
        with open(osp.join(self.data_root, f"annotations/dexycb_bone_length.json")) as f:
            self.bone_length = json.load(f)["bone"]
            print("load bone length from", osp.join(self.data_root, f"annotations/dexycb_bone_length.json"))
        self.color_format = "color_{:06d}.jpg"
        self.label_format = "labels_{:06d}.npz"
        self.height = 480
        self.width = 640
        subject_ind, serial_ind, sequence_ind = get_data_split(self.data_split, setup=setup)
        self.subjects = [_SUBJECTS[i] for i in subject_ind]        
        self.serials = [_SERIALS[i] for i in serial_ind]

        self.cache_folder = os.path.join(CACHE_HOME, self.name)
        os.makedirs(self.cache_folder, exist_ok=True)        
        cache_filename = f"{self.data_split}_seq_T{self.seqlen}_S{self.stride}" if self.seq_load else self.data_split
        if setup != 's1':
            cache_filename += f"_{setup}"
        cache_filename += ".pkl"
        cache_path = os.path.join(self.cache_folder, cache_filename)            
        if os.path.exists(cache_path) and use_cache:
            print(f"loading from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                data_list_dict = pickle.load(f)
                self.sequences = data_list_dict["sequences"]
                self.sv_data = data_list_dict["data"]
                self.vid_data = data_list_dict["vid_indices"]
                self.ycb_ids = data_list_dict["ycb_ids"]
                self.ycb_grasp_ind = data_list_dict["ycb_grasp_ind"]
                self.mano_side = data_list_dict["mano_side"]
                self.mano_betas = data_list_dict["mano_betas"]
        else:
            self.vid_data = []
            self.sequences = []
            self.sv_data = []
            self.ycb_ids = []
            self.ycb_grasp_ind = []
            self.mano_side = []
            self.mano_betas = []
            offset = 0
            for n in tqdm(self.subjects):
                seq = sorted([path for path in os.listdir(os.path.join(self.data_root, n))])
                seq = [os.path.join(n, s) for s in seq if os.path.isdir(os.path.join(self.data_root, n, s))]
                assert len(seq) == 100, f"{len(seq)}, {seq}"
                seq = [seq[i] for i in sequence_ind]
                self.sequences += seq
                for sub_id, q in enumerate(seq):
                    meta_file = os.path.join(self.data_root, q, "meta.yml")
                    with open(meta_file, 'r') as f:
                        meta = yaml.load(f, Loader=yaml.FullLoader)
                    c = np.arange(len(self.serials))
                    f = np.arange(meta['num_frames'])
                    f, c = np.meshgrid(f, c)
                    c = c.ravel()
                    f = f.ravel()
                    s = (offset + sub_id) * np.ones_like(c)
                    m = np.vstack((s, c, f)).T      
                    self.sv_data.append(m)

                    # Setting up vid_indices for current subject-sequence-camera triple
                    if self.seq_load:
                        for cam_id in range(len(self.serials)):           
                            # Initiate sequences starting from the first valid frame
                            start_frame_idx = 0
                            while (start_frame_idx < meta['num_frames'] - seqlen + 1):
                                # check vis for init frame
                                label_path = os.path.join(self.data_root, q, self.serials[cam_id], self.label_format.format(start_frame_idx))
                                assert os.path.exists(label_path), f"{label_path}"
                                vis = self._visibility_check(label_path)
                                # if vis.sum() != self.njoints: # check all visible at the 1st frame
                                if vis[CENTER_ID] == 0 or vis.sum() < N_VALID_KEYPOINTS: # skip if root is not visible or limited visibility
                                    start_frame_idx += 1 # shift by 1
                                    continue
                                end_frame_idx = start_frame_idx + seqlen - 1
                                if end_frame_idx >= meta['num_frames']:
                                    break
                                self.vid_data.append((s[0], cam_id, sub_id, (start_frame_idx, end_frame_idx)))
                                start_frame_idx += stride # shift by stride
                            # print(self.vid_data[:3], self.vid_data[-3:], len(self.vid_data), meta['num_frames'])
                            # exit()
                                
                    self.ycb_ids.append(meta['ycb_ids'])
                    self.ycb_grasp_ind.append(meta['ycb_grasp_ind'])
                    self.mano_side.append(meta['mano_sides'][0])
                    mano_calib_file = os.path.join(self.data_root, "calibration", "mano_{}".format(meta['mano_calib'][0]), "mano.yml")
                    with open(mano_calib_file, 'r') as f:
                        mano_calib = yaml.load(f, Loader=yaml.FullLoader)
                    self.mano_betas.append(mano_calib['betas'])
                offset += len(seq)
            self.sv_data = np.vstack(self.sv_data)
            
            # invalid_list = []
            if self.seq_load: # remove invalid seqs
                print("checking invalid seqs")
                data_idx_list = list(range(len(self.vid_data)))[::-1] # reverse order
                # vid_indices = deepcopy(self.vid_data)
                for idx in tqdm(data_idx_list):
                    # GT vis check for a seq
                    s, c, _, frame_range = self.vid_data[idx]
                    vis = self._seq_visibility_check(s, c, frame_range)
                    init_vis = vis[0] #.sum()
                    vis_per_frame = sum([v.sum() for v in vis])/len(vis)
                    if vis_per_frame < N_VALID_KEYPOINTS:
                        s, c, _, _ = self.vid_data[idx]    
                        print(f"exclude {idx}th seq: s: {s}, c: {c}, vis_per_frame: {vis_per_frame:.1f}")
                        # invalid_list.append(idx)
                        self.vid_data.pop(idx)
                        continue
                    # pred vis check
                    seq_sample = self.get_seq_sample(idx)
                    pred_det_mask = seq_sample["pred_det_mask"]
                    if pred_det_mask.sum() < VALID_SEQ_SAMPLE_RATE * len(pred_det_mask) or pred_det_mask[0] == 0:
                        print(f"exclude {idx}th seq: s: {s}, c: {c}, pred_det_mask: {pred_det_mask.sum() / len(pred_det_mask):.1f}, pred_det_mask[0]: {pred_det_mask[0]}")
                        # invalid_list.append(idx)
                        self.vid_data.pop(idx)
                print(f"#invalid seqs: {(len(data_idx_list) - len(self.vid_data))}")
                # assert len(invalid_list) == (len(data_idx_list) - len(self.vid_data)), f"{len(invalid_list)}, {len(data_idx_list)}, {len(self.vid_data)}"
            else: # remove invalid frames                
                print("checking invalid frames")
                data_idx_list = list(range(len(self.sv_data)))[::-1] # reverse order
                for idx in tqdm(data_idx_list):
                    (_s, _c, _f) = self._map_index_to_frame_info(idx)
                    vis = self._visibility_check((_s, _c, _f))
                    # n_kpt and root check
                    if vis.sum() < N_VALID_KEYPOINTS or vis[self.ref_bone_link[0]] == 0 or vis[self.ref_bone_link[1]] == 0:
                        # invalid_list.append(idx)
                        self.sv_data = np.delete(self.sv_data, idx, axis=0)
                print(f"#invalid imagess: {len(data_idx_list) - len(self.sv_data)}")
                # for idx in invalid_list[::-1]:
                    # self.sv_data = np.delete(self.sv_data, idx, axis=0)        
            if use_cache:
                print(f"Saving cache for split {self.data_split} to {cache_path}")
                data_list_dict = {
                    "sequences": self.sequences,
                    "data": self.sv_data,
                    "vid_indices": self.vid_data,
                    "ycb_ids": self.ycb_ids,
                    "ycb_grasp_ind": self.ycb_grasp_ind,
                    "mano_side": self.mano_side,
                    "mano_betas": self.mano_betas,
                }
                with open(cache_path, "wb") as fid:
                    pickle.dump(data_list_dict, fid)
                
        print(f"loading dataset (setup: {setup}): [data_split: {self.data_split:<5}][# images: {len(self.sv_data):>8,}][# seq (T={self.seqlen},S={self.stride}): {len(self.vid_data):8,}]")

    def set_pose_cache(self, pose_cache_path):
        assert os.path.exists(pose_cache_path), f"{pose_cache_path} not found"   
        self.pred_pts_3d_dict = torch.load(pose_cache_path)
        # self.gt_pts_3d_dict = torch.load(pose_cache_path.replace("pred", "gt"))
        self.gt_pts_3d_dict = None
        self.pose_cache_path = pose_cache_path
    
    def __len__(self):
        if self.seq_load:
            return len(self.vid_data)
        else:
            return len(self.sv_data)
        
    def _set_sample_info(self, s, c, f):
        d = os.path.join(self.data_root, self.sequences[s], self.serials[c])
        sample = {
            'color_file': os.path.join(d, self.color_format.format(f)),
            'depth_file': None,
            'label_file': os.path.join(d, self.label_format.format(f)),
            'ycb_ids': self.ycb_ids[s],
            'ycb_grasp_ind': self.ycb_grasp_ind[s],
            'mano_side': self.mano_side[s],
            'mano_betas': self.mano_betas[s],
            'cam_id': CAM2ID[self.serials[c]],
            'sub_id': SUB2ID[self.sequences[s].split("/")[0]],
        }        
        return sample
    
    def _map_index_to_frame_info(self, index):
        s, c, f = self.sv_data[index]
        return (s, c, f)
        
    def __getitem__(self, idx):
        s, c, f = self._map_index_to_frame_info(index)
        return self._set_sample_info(s, c, f)
    
    def get_seq_sample(self, index):
        """Retrieve a sequence of frames (a video chunk)"""
        s, c, _, (start_index, end_index) = self.vid_data[index]

        # Use get_sample to retrieve each frame sample in the sequence
        try:
            samples = [self._get_frame_sample(s, c, f) for f in range(start_index, end_index+1)]
        except:
            raise ValueError(f"{self.vid_data[index]}")
        
        # Merge lists of dictionaries into a single dictionary with lists as values
        seq_sample = {k: [d[k] for d in samples] for k in samples[0]}
        if self.pose_cache_path is not None:
            self._set_processed_seq_sample(s, c, (start_index, end_index), seq_sample)
            
        return seq_sample
    
    def _set_processed_seq_sample(self, s, c, frame_range, seq_sample={}):
        sub_name, capture_id = self.sequences[s].split("/")
        sub_id = SUB2ID[sub_name]
        cam_id = CAM2ID[self.serials[c]]
        start_index, end_index = frame_range
        hand_type = self.mano_side[s]

        img_path_list = seq_sample["img_path"]            
        gt_pts_3d_cam = torch.from_numpy(np.stack(seq_sample["joint"], axis=0))
        pred_pts_3d_cam = np.zeros((self.seqlen, 21, 3))
        pred_det_mask = np.zeros((self.seqlen))
        if capture_id in self.pred_pts_3d_dict[sub_id][cam_id].keys():
            # pred_pts_3d_cam = [self.pred_pts_3d_dict[sub_id][cam_id][capture_id][img_path] for img_path in img_path_list]
            for i, img_path in enumerate(img_path_list):
                # assert os.path.exists(img_path), f"{img_path}"
                if img_path in self.pred_pts_3d_dict[sub_id][cam_id][capture_id].keys():
                    pred_pts_3d_cam[i] = self.pred_pts_3d_dict[sub_id][cam_id][capture_id][img_path]
                    pred_det_mask[i] = 1
            pred_pts_3d_cam = torch.from_numpy(pred_pts_3d_cam)
            pred_pts_3d_cam = align.global_align(gt_pts_3d_cam, pred_pts_3d_cam)[1]
            
            # compute joint bone
            if SCALE_NORM == "per_seq":
                bone_length = np.array(self.bone_length[sub_name][capture_id]["avg"]) # mm scale  
                joint_bone = np.ones((self.seqlen, 1)) * bone_length / 1000.   # m scale     
            elif SCALE_NORM == "per_frame":
                # scale norm
                joint_bone = torch.zeros((self.seqlen, 1))
                for jid, nextjid in zip(self.ref_bone_link[:-1], self.ref_bone_link[1:]):
                    joint_bone += torch.norm(gt_pts_3d_cam[:, nextjid] - gt_pts_3d_cam[:, jid], dim=1, keepdim=True)
            
            # pred_pts_3d_cam = pred_pts_3d_cam / (joint_bone[:, None, :] + 1e-8)                        
            pred_pts_3d_cam = pred_pts_3d_cam.numpy()        
            gt_pts_3d_cam = gt_pts_3d_cam.numpy()
    
        if self.unlabeled and self.data_split == "train":
            gt_pts_3d_cam = pred_pts_3d_cam.copy()
            seq_sample["vis"] = vis_pred
        # extend det_mask to joint-wise vis_pred
        vis_pred = np.tile(pred_det_mask[:, None], (1, 21))
        seq_sample.update({
            "pred_pts_3d": pred_pts_3d_cam,
            # "pred_det_mask": pred_det_mask, 
            "vis_pred": vis_pred,
            "joint": gt_pts_3d_cam,
            "joint_bone": joint_bone,
            "start_frame_idx": start_index,
            "end_frame_idx": end_index,
        })
        return seq_sample
                
    def get_sample(self, index): # per-frame sample
        (s, c, f) = self._map_index_to_frame_info(index)
        frame_sample = self._get_frame_sample(s, c, f)            
        frame_sample["index"] = index
        return frame_sample
    
    def _visibility_check(self, input):
        if isinstance(input, tuple): # (s, c, f)
            assert len(input) == 3
            (s, c, f) = input
            input = self._set_sample_info(s, c, f)
            
        if isinstance(input, dict): # sample
            labels = np.load(input["label_file"]) 
        elif isinstance(input, str): # label_path
            assert os.path.exists(input), f"{input}"
            labels = np.load(input)
        else:
            raise NotImplementedError()
        
        pts_2d = labels["joint_2d"][0] # (21, 2), in pixel
        pts_3d = labels["joint_3d"][0] # (21, 3), in meter img_camera
        assert pts_2d.shape == (21, 2) and pts_3d.shape == (21, 3), f"pts_2d: {pts_2d.shape}, pts_3d: {pts_3d.shape}"
        vis = (pts_2d[:, 0] != -1) & (pts_2d.min(axis=1) > 0) & (pts_2d[:, 0] <= self.width) & (pts_2d[:, 1] <= self.height) & (pts_3d[:, 0] != -1)
        return vis
    
    def _seq_visibility_check(self, s, c, frame_range):
        #s, c, _, (start_index, end_index) = self.vid_data[index]
        start_index, end_index = frame_range
        vis = [self._visibility_check((s, c, f)) for f in range(start_index, end_index+1)]
        return vis
    
    def _get_frame_sample(self, s, c, f):
        # 1 get data from the original dataset
        sample = self._set_sample_info(s, c, f)
        img_path = sample["color_file"]
        clr = None
        if not self.no_img_load:
            clr = Image.open(sample["color_file"]).convert("RGB")
        hand_type = sample["mano_side"]
        sub_id = sample["sub_id"]
        cam_id = sample["cam_id"]
        capture_id = img_path.split(f"{ID2SUB[sub_id]}/")[1].split("/")[0]
        
        K = np.array(self.calib['intrinsics'][ID2CAM[cam_id]])
        RT = np.array(self.calib['extrinsics'][ID2SUB[sub_id]][capture_id][ID2CAM[cam_id]])
        
        # 2 get 2d kp in img space
        labels = np.load(sample["label_file"]) 
        pts_2d = labels["joint_2d"][0] # (21, 2), in pixel
        pts_3d = labels["joint_3d"][0] # (21, 3), in meter img_camera
        vis =  self._visibility_check(sample)
        
        # 3 handle the left-right hand problem
        # a sample has right/left "hand_type", do flip if hand_type != hand_side
        if hand_type != self.hand_side: # flip
            if not self.no_img_load:
                clr = clr.transpose(Image.FLIP_LEFT_RIGHT)
                assert self.width == clr.size[0], f"{self.width}, {clr.size[0]}"
            center = handutils.get_annot_center(pts_2d)            
            center[0] = self.width - center[0]
            pts_2d[:, 0] = self.width - pts_2d[:, 0]
            pts_3d[:, 0] *= -1
        
        # 4 calculate the parameters for cropping from 2dkp
        center = handutils.get_annot_center(pts_2d)
        
        crop_scale = handutils.get_ori_crop_scale(mask=None, side=None, mask_flag=False, pts_2d=pts_2d, scale_factor=1.75)
        
        frame_sample = {
            'img_path': img_path,
            'sub_id': int(sub_id),
            'cam_id': int(cam_id),
            'capture_id': capture_id,
            # @2024.05.10 exchange pts_2d to kp2d
            'kp2d': pts_2d,
            'center': center,
            # @2024.05.10 exchange crop_scale to my_scale
            'my_scale': crop_scale,
            'joint': pts_3d,
            'vis': vis,
            'K': K,
            'RT': RT,
            'hand_id': HAND2ID[hand_type],
        }
        if not self.no_img_load:
            frame_sample['clr'] = clr
        
        if self.vis:
            vis_kpt_3d(clr, pts_2d, pts_3d)

        return frame_sample
    
if __name__ == "__main__":
    import sys
    print(sys.path)
    data_root = "/Users/take/Backup/dexycb" #"data/DexYCB"
    dataset = DexYCBDataset(
        data_root=data_root,
        data_split='val',
        hand_side='right',
        njoints=21,        
        seq_load=True,
        seqlen=32, 
        stride=32,
        vis=True,
        is_debug=True,
        pose_cache_path=".cache/preds/230908_070035_detnet_train_dy_ckp_detnet_010/pred_pose_3d_val_dy.pth",
    )
    print(dataset.vid_indices[:5])
    # TODO: check vis chunk 
    
    # sample = dataset.get_sample(100)
    # sample = dataset.get_seq_sample(100)    
    # print(len(sample["clr"]))
    # print(sample["index"])
    for idx in range(len(dataset)):
        sample = dataset.get_seq_sample(idx)
        n_vis = sum([vis.sum() for vis in sample["vis"]])/21.
        print(idx, n_vis, [vis.sum() for vis in sample["vis"]])
        # if n_vis < 5:
        #     print(idx, n_vis, dataset.vid_indices[idx])
    # print(dataset.vid_indices[:5])

    
    """
    only with vis check for gt data    
    n_total_samples: 1378, n_detected_samples: 1120, n_undetected_samples: 28
    + with seq pred check    
    n_total_samples: 22650, n_detected_samples: 22612, n_undetected_samples: 44
    01 21:11 - INFO - eval result (set: dy): {
        "mpjpe-rel": 16.42,
        "mpjpe-abs": 11.76,
        "mpjpe-2d-rel": 12.98,
        "mpjpe-2d-abs": 9.3,
        "mpjpe-z-rel": 8.25,
        "mpjpe-z-abs": 5.85,
        "pck-auc-rel": 69.7,
        "acc-error": 7.18,
        "mpjpe-rel-det": 16.33,
        "mpjpe-rel-undet": 101.34,
        "n_detected": 22612.0,
        "n_undetected": 44.0
    }
    """
