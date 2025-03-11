# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
from scipy import linalg

def cam2pixel(cam_coord, input):
    torch_input = False
    if isinstance(input, tuple):
        f, c = input
        if isinstance(f[0], torch.Tensor):
            f = np.array(f)
            c = np.array(c)
            torch_input = True
    elif input.shape == (3, 3):
        if isinstance(input, torch.Tensor):
            input = input.numpy()
            torch_input = True
        f = [input[0, 0], input[1, 1]]
        c = [input[0, 2], input[1, 2]]
    else:
        raise ValueError('input type not supported')
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]), 1)
    img_coord = torch.from_numpy(img_coord).float() if torch_input else img_coord
    return img_coord

def pixel2cam(pixel_coord, input):
    torch_input = False
    if isinstance(input, tuple):
        f, c = input
        if isinstance(f[0], torch.Tensor):
            f = np.array(f)
            c = np.array(c)
            torch_input = True
    elif input.shape == (3, 3):
        if isinstance(input, torch.Tensor):
            input = input.numpy()
            torch_input = True
        f = [input[0, 0], input[1, 1]]
        c = [input[0, 2], input[1, 2]]
        torch_input = True
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]), 1)
    cam_coord = torch.from_numpy(cam_coord).float() if torch_input else cam_coord
    return cam_coord

def world2cam_interhand(world_coord, R, t):
    torch_input = False
    if isinstance(world_coord, torch.Tensor):
        world_coord = world_coord.numpy()
        torch_input = True
    cam_coord = np.dot(R, world_coord - t)
    cam_coord = torch.from_numpy(cam_coord).float() if torch_input else cam_coord
    return cam_coord

def world2cam(pts_3d, R, t):
    # pts_3d: Jx3, R: 3x3, t: 3
    torch_input = batch_input = False
    if isinstance(pts_3d, torch.Tensor):
        pts_3d = pts_3d.numpy()
        torch_input = True        
    if pts_3d.ndim == 3:  # Single input
        n_joints = pts_3d.shape[1]
        pts_3d = pts_3d.reshape(-1, 3)
        batch_input = True        
    pts_cam = np.dot(R, pts_3d.T).T + t
    pts_cam = pts_cam.reshape(-1, n_joints, 3) if batch_input else pts_cam
    pts_cam = torch.from_numpy(pts_cam).float() if torch_input else pts_cam
    return pts_cam

def cam2world(pts_cam_3d, R, t):
    # pts_3d: Jx3, R: 3x3, t: 3
    torch_input = batch_input = False
    if isinstance(pts_cam_3d, torch.Tensor):
        pts_cam_3d = pts_cam_3d.numpy()
        torch_input = True
    inv_R = np.linalg.inv(R)
    if pts_cam_3d.ndim == 3:
        n_joints = pts_cam_3d.shape[1]
        pts_cam_3d = pts_cam_3d.reshape(-1, 3)
        batch_input = True    
    pts_3d = np.dot(inv_R, (pts_cam_3d - t).T).T    
    pts_3d = pts_3d.reshape(-1, n_joints, 3) if batch_input else pts_3d
    pts_3d = torch.from_numpy(pts_3d).float() if torch_input else pts_3d
    return pts_3d

def world2pixel(pts_3d, KRT):
    assert pts_3d.shape[1] == 3, f"shape error: {pts_3d.shape}"
    torch_input = False
    if isinstance(pts_3d, torch.Tensor):
        pts_3d = pts_3d.numpy()
        torch_input = True
    _pts_3d = np.concatenate((pts_3d[:, :3], np.ones((pts_3d.shape[0], 1))), axis=-1)
    pts_2d = np.matmul(_pts_3d, KRT.T)
    pts_2d /= pts_2d[:, 2:3]
    pts_2d = torch.from_numpy(pts_2d).float() if torch_input else pts_2d
    return pts_2d

def multi_meshgrid(*args):
    """
    Creates a meshgrid from possibly many
    elements (instead of only 2).
    Returns a nd tensor with as many dimensions
    as there are arguments
    """
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)


def flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
                            dtype=torch.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    assert flipped.device == tensor.device
    assert flipped.requires_grad == tensor.requires_grad
    return flipped

class Camera(object):
    def __init__(self, K, Rt, dist=None, name=""):
        # Rotate first then translate
        self.K = np.array(K).copy()
        assert self.K.shape == (3, 3)

        self.Rt = np.array(Rt).copy()
        assert self.Rt.shape == (3, 4)

        self.dist = dist
        if self.dist is not None:
            self.dist = np.array(self.dist).copy().flatten()

        self.name = name

    def update_after_crop(self, bbox):
        left, upper, right, lower = bbox

        cx, cy = self.K[0, 2], self.K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[0, 2], self.K[1, 2] = new_cx, new_cy

    def update_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_height, new_width = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = new_fx, new_fy, new_cx, new_cy

    @property
    def projection(self):
        return np.dot(self.K, self.Rt)

    def factor(self):
        """  Factorize the camera matrix into K,R,t as P = K[R|t]. """

        # factor first 3*3 part
        K,R = linalg.rq(self.projection[:, :3])

        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T) < 0:
            T[1,1] *= -1

        K = np.dot(K,T)
        R = np.dot(T,R) # T is its own inverse
        t = np.dot(linalg.inv(self.K), self.projection[:,3])

        return K, R, t

    def get_params(self):
        K, R, t = self.factor()
        campos, camrot = t, R
        focal = [K[0, 0], K[1, 1]]
        princpt = [K[0, 2], K[1, 2]]
        return campos, camrot, focal, princpt


