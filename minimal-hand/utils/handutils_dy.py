import numpy as np
import torch

try:
    from PIL import Image
except ImportError:
    print('Could not import PIL in handutils')

DEPTH_RANGE = 3.0
DEPTH_MIN = -1.5

def get_joint_bone(joint, ref_bone_link=None):
    """
    Compute the bone length for each instance in a batch of joint coordinates.
    
    Parameters:
    - joint (Tensor or ndarray): Shape (B, 21, 3) representing B instances of joint coordinates.
    - ref_bone_link (tuple): Containing the joint indices to compute the bone length for.

    Returns:
    - Tensor or ndarray: Shape (B, 1) with the computed bone length for each instance.
    """
    if ref_bone_link is None:
        ref_bone_link = (0, 9)

    if (
            not torch.is_tensor(joint)
            and not isinstance(joint, np.ndarray)
    ):
        raise TypeError('joint should be ndarray or torch tensor. Got {}'.format(type(joint)))
    if (
            len(joint.shape) != 3
            or joint.shape[1] != 21
            or joint.shape[2] != 3
    ):
        raise TypeError('joint should have shape (B, njoint, 3), Got {}'.format(joint.shape))

    batch_size = joint.shape[0]
    bone = 0
    if torch.is_tensor(joint):
        bone = torch.zeros((batch_size, 1)).to(joint.device)
        for jid, nextjid in zip(
                ref_bone_link[:-1], ref_bone_link[1:]
        ):
            bone += torch.norm(
                joint[:, jid, :] - joint[:, nextjid, :],
                dim=1, keepdim=True
            )  # (B, 1)
    elif isinstance(joint, np.ndarray):
        bone = np.zeros((batch_size, 1))
        for jid, nextjid in zip(
                ref_bone_link[:-1], ref_bone_link[1:]
        ):
            bone += np.linalg.norm(
                (joint[:, jid, :] - joint[:, nextjid, :]),
                ord=2, axis=1, keepdims=True
            )  # (B, 1)
    return bone


def uvd2xyz(
        uvd,
        joint_root,
        joint_bone,
        intr=None,
        trans=None,
        scale=None,
        inp_res=256,
        mode='persp'
):
    """
    Convert from UVD (pixel, pixel, depth) coordinates to XYZ (3D space) coordinates.
    
    Parameters:
    - uvd (Tensor): Shape (B, M, 3) representing B instances of 2D hand joint locations + depth.
    - joint_root (int): Index of the root joint.
    - joint_bone (float): Length of the reference bone.
    - intr (Tensor or None): Camera intrinsic parameters. Shape (3,3).
    - trans (Tensor or None): Translation vector. Shape (3,).
    - scale (float or None): Scaling factor.
    - inp_res (int): Input resolution.
    - mode (str): Perspective mode. Can be 'persp' or 'ortho'.

    Returns:
    - Tensor: Shape (B, M, 3) representing the 3D coordinates.
    """
    bs = uvd.shape[0]
    if mode in ['persp', 'perspective']:
        if intr is None:
            raise Exception("No intr found in perspective")
        '''1. denormalized uvd'''
        uv = uvd[:, :, :2] * inp_res  # 0~256
        depth = (uvd[:, :, 2] * DEPTH_RANGE) + DEPTH_MIN
        root_depth = joint_root[:, -1].unsqueeze(-1)  # (B, 1)
        z = depth * joint_bone.expand_as(uvd[:, :, 2]) + \
            root_depth.expand_as(uvd[:, :, 2])  # B x M

        '''2. uvd->xyz'''
        camparam = torch.zeros((bs, 4)).float().to(intr.device)  # (B, 4)
        camparam[:, 0] = intr[:, 0, 0]  # fx
        camparam[:, 1] = intr[:, 1, 1]  # fx
        camparam[:, 2] = intr[:, 0, 2]  # cx
        camparam[:, 3] = intr[:, 1, 2]  # cy
        camparam = camparam.unsqueeze(1).expand(-1, uvd.size(1), -1)  # B x M x 4
        xy = ((uv - camparam[:, :, 2:4]) / camparam[:, :, :2]) * \
             z.unsqueeze(-1).expand_as(uv)  # B x M x 2
        return torch.cat((xy, z.unsqueeze(-1)), -1)  # B x M x 3
    elif mode in ['ortho', 'orthogonal']:
        if trans is None or scale is None:
            raise Exception("No trans or scale found in orthorgnal")
        raise Exception("orth Unimplement !")
    else:
        raise Exception("Unkonwn mode type. should in ['persp', 'ortho']")


def xyz2uvd(
        xyz,
        joint_root,
        joint_bone,
        intr=None,
        trans=None,
        scale=None,
        inp_res=256,
        mode='persp'
):
    """
    Convert from XYZ (3D space) coordinates to UVD (pixel, pixel, depth) coordinates.
    
    Parameters:
    - xyz (Tensor): Shape (B, M, 3) representing B instances of 3D hand joint locations.
    - joint_root (int): Index of the root joint.
    - joint_bone (float): Length of the reference bone.
    - intr (Tensor or None): Camera intrinsic parameters. Shape (3,3).
    - trans (Tensor or None): Translation vector. Shape (3,).
    - scale (float or None): Scaling factor.
    - inp_res (int, optional): Input resolution. Defaults to 256.
    - mode (str, optional): Perspective mode. Can be 'persp' or 'ortho'. Defaults to 'persp'.
    
    Returns:
    - Tensor: Shape (B, M, 3) representing the UVD coordinates.
    """
    bs = xyz.shape[0]
    if mode in ['persp', 'perspective']:
        if intr is None:
            raise Exception("No intr found in perspective")
        z = xyz[:, :, 2]
        xy = xyz[:, :, :2]
        xy = xy / z.unsqueeze(-1).expand_as(xy)

        ''' 1. normalize depth : root_relative, scale_invariant '''
        root_depth = joint_root[:, -1].unsqueeze(-1)  # (B, 1)
        depth = (z - root_depth.expand_as(z)) / joint_bone.expand_as(z)

        '''2. xy->uv'''
        camparam = torch.zeros((bs, 4)).float().to(intr.device)  # (B, 4)
        camparam[:, 0] = intr[:, 0, 0]  # fx
        camparam[:, 1] = intr[:, 1, 1]  # fx
        camparam[:, 2] = intr[:, 0, 2]  # cx
        camparam[:, 3] = intr[:, 1, 2]  # cy
        camparam = camparam.unsqueeze(1).expand(-1, xyz.size(1), -1)  # B x M x 4
        uv = (xy * camparam[:, :, :2]) + camparam[:, :, 2:4]

        '''3. normalize uvd to 0~1'''
        uv = uv / inp_res
        depth = (depth - DEPTH_MIN) / DEPTH_RANGE

        return torch.cat((uv, depth.unsqueeze(-1)), -1)
    elif mode in ['ortho', 'orthogonal']:
        if trans is None or scale is None:
            raise Exception("No trans or scale found in orthorgnal")
        raise Exception("orth Unimplement !")
    else:
        raise Exception("Unkonwn proj type. should in ['persp', 'ortho']")


def persp_joint2kp(joint, intr):
    """
    Convert 3D joint coordinates to 2D keypoint coordinates using camera intrinsic parameters.

    Parameters:
    - joint (Tensor): Shape (B, M, 3) representing B instances of 3D joint coordinates.
    - intr (Tensor): Camera intrinsic parameters. Shape (3,3).

    Returns:
    - Tensor: Shape (B, M, 2) representing the 2D keypoint coordinates.
    """
    joint_homo = torch.matmul(joint, intr.transpose(1, 2))
    pts_2d = joint_homo / joint_homo[:, :, 2:]
    pts_2d = pts_2d[:, :, :2]
    return pts_2d


def rot_pts_2d(pts_2d, rot):
    """
    Rotate 2D keypoints based on a given rotation matrix.
    
    Parameters:
    - pts_2d (Tensor): Shape (B, M, 2) representing B instances of 2D keypoints.
    - rot (Tensor): Rotation matrix or tensor for transforming the keypoints. Expected shape would be (2, 2) or (B, 2, 2) depending on the implementation.

    Returns:
    - Tensor: Rotated 2D keypoints with the same shape as `pts_2d`.
    """    
    pts_2d = np.concatenate((pts_2d, np.ones((pts_2d.shape[0], 1))), axis=1)
    new_pts_2d = np.matmul(pts_2d, rot.transpose())
    return new_pts_2d


def get_annot_scale(annots, visibility=None, scale_factor=2.0):
    """
    Retrieves the size of the square to crop around the hand annotation. It 
    computes the size by taking the maximum of the vertical and horizontal 
    span of the hand and multiplying it with a scale factor.
    
    Parameters:
        annots (np.ndarray): Hand annotations.
        visibility (list or np.ndarray, optional): Visibility list for annotations.
        scale_factor (float, optional): Multiplier for padding. Default is 2.0.

    Returns:
        float: Computed size for the square to crop.
    """
    if visibility is not None:
        annots = annots[visibility]
    min_x, min_y = annots.min(0)
    max_x, max_y = annots.max(0)
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    max_delta = max(delta_x, delta_y)
    s = max_delta * scale_factor
    return s


def get_mask_mini_scale(mask_, side):
    """
    Retrieves the size of the square to encapsulate either the left or right 
    side of the mask or the entire mask, based on the side parameter provided.
    
    Parameters:
        mask_ (np.ndarray): The mask to compute scale for.
        side (str or int): Determines which side of the mask to process.
            Accepts 'l' for left, 'r' for right, or 0 for the whole mask.

    Returns:
        int: Computed mask scale.

    Raises:
        ValueError: If the computed mask_scale is 0.
    """
    # mask = np.array(mask_.copy())[:, :, 2:].squeeze()
    mask = mask_.copy().squeeze()
    mask_scale = 0
    # print(mask.shape)
    if side == "l":
        id_left = [i for i in range(2, 18)]
        np.putmask(mask, np.logical_and(mask >= id_left[0], mask <= id_left[-1]), 128)
        seg = np.argwhere(mask == 128)
        # print("seg.shape=",seg.shape)
        seg_rmin, seg_cmin = np.min(seg, axis=0)
        seg_rmax, seg_cmax = np.max(seg, axis=0)
        mask_scale = max(seg_rmax - seg_rmin + 1, seg_cmax - seg_cmin + 1)

    elif side == "r":
        id_right = [i for i in range(18, 34)]
        np.putmask(mask, np.logical_and(mask >= id_right[0], mask <= id_right[-1]), 255)

        seg = np.argwhere(mask == 255)
        seg_rmin, seg_cmin = np.min(seg, axis=0)
        seg_rmax, seg_cmax = np.max(seg, axis=0)
        mask_scale = max(seg_rmax - seg_rmin + 1, seg_cmax - seg_cmin + 1)
    elif side == 0:
        rmin, cmin = mask.min(0)
        rmax, cmax = mask.max(0)
        mask_scale = max(rmax - rmin + 1, cmax - cmin + 1)

    if not mask_scale:
        raise ValueError("mask_scale is 0!")

    return mask_scale


def get_pts_2d_mini_scale(annots):
    """
    Computes the minimum square size required to include all 2D keypoints.
    
    Parameters:
        annots (np.ndarray): 2D keypoints.

    Returns:
        float: Minimum scale size.
    """
    min_x, min_y = annots.min(0)  # opencv convention
    max_x, max_y = annots.max(0)

    delta_x = max_x - min_x
    delta_y = max_y - min_y

    max_delta = max(delta_x, delta_y)

    return max_delta

def get_ori_crop_scale(mask, side, pts_2d, mask_flag=True, scale_factor=2.0):
    """
    Computes the original cropping scale based on the mask and 2D keypoints.
    
    Parameters:
        mask (np.ndarray): The mask to compute scale for.
        side (str or int): Determines which side of the mask to process.
            Accepts 'l' for left, 'r' for right, or 0 for the whole mask.
        pts_2d (np.ndarray): 2D keypoints.
        mask_flag (bool, optional): Indicates if mask is to be considered. Default is True.
        scale_factor (float, optional): Scale factor for padding. Default is 2.0.

    Returns:
        float: Original cropping scale.
    """
    pts_2d_mini_scale = get_pts_2d_mini_scale(pts_2d)

    ori_crop_scale =pts_2d_mini_scale

    # if mask.any()!=None:
    if mask_flag:
        # print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
        mask_mini_scale = get_mask_mini_scale(mask, side)
        ori_crop_scale = max(mask_mini_scale, pts_2d_mini_scale)

    # if ori_crop_scale % 2 == 0:
    #     ori_crop_scale += 2
    # else:
    #     ori_crop_scale += 3

    return ori_crop_scale * scale_factor

def get_annot_center(annots, visibility=None):
    """
    Compute the center point of given hand annotations.
    
    Parameters:
        annots (np.ndarray): Hand annotations.
        visibility (list or np.ndarray, optional): Visibility list for annotations.
        
    Returns:
        np.ndarray: Array containing x and y coordinates of the center point.
    """
    # Get scale
    if visibility is not None:
        annots = annots[visibility]
    min_x, min_y = annots.min(0)
    max_x, max_y = annots.max(0)
    c_x = int((max_x + min_x) / 2)
    c_y = int((max_y + min_y) / 2)
    return np.asarray([c_x, c_y])


def transform_coords(pts, affine_trans, invert=False):
    """
    Transforms the given 2D points using the specified affine transformation matrix.
    
    Parameters:
        pts (np.ndarray): Points to be transformed in the shape (point_nb, 2).
        affine_trans (np.ndarray): Affine transformation matrix.
        invert (bool, optional): If set to True, inverts the affine transformation matrix. Default is False.

    Returns:
        np.ndarray: Transformed points.
    """
    if invert:
        affine_trans = np.linalg.inv(affine_trans)
    hom2d = np.concatenate([pts, np.ones([np.array(pts).shape[0], 1])], 1)
    transformed_rows = affine_trans.dot(hom2d.transpose()).transpose()[:, :2]
    return transformed_rows.astype(int)

def transform_coords_batch(pts, affine_trans, invert=False):
    """
    Batch processing of transform_coords
    
    Parameters:
        pts (np.ndarray): Points to be transformed in the shape (batch_size, point_nb, 2).
        affine_trans (np.ndarray): Affine transformation matrices in the shape (batch_size, 3, 3).
        invert (bool, optional): If set to True, inverts the affine transformation matrices. Default is False.

    Returns:
        np.ndarray: Transformed points in the shape (batch_size, point_nb, 2).
    """
    batch_size = pts.shape[0]
    if invert:
        affine_trans = np.linalg.inv(affine_trans)
    
    # Convert to homogeneous coordinates
    hom2d = np.concatenate([pts, np.ones((batch_size, pts.shape[1], 1))], axis=2)
    
    # Apply affine transformation
    transformed = np.matmul(hom2d, affine_trans.transpose(0, 2, 1))
    
    return transformed[:, :, :2].astype(int)


def transform_img(img, affine_trans, res):
    """
    Transforms an image using the specified affine transformation matrix.
    
    Parameters:
        img (PIL.Image): The source image.
        affine_trans (np.ndarray): Affine transformation matrix.
        res (tuple): Final image size.

    Returns:
        PIL.Image: Transformed image.
    """
    trans = np.linalg.inv(affine_trans)

    img = img.transform(
        tuple(res), Image.AFFINE, (trans[0, 0], trans[0, 1], trans[0, 2],
                                    trans[1, 0], trans[1, 1], trans[1, 2])
    )
    return img


def get_affine_transform(center, scale, optical_center, out_res, rot=0):
    """
    Computes an affine transformation matrix based on given parameters. Also, 
    it provides post rotation transformation. Note that an image is rotated 
    around optical center, not the image center.
    
    Parameters:
        center (np.ndarray or list): Center point.
        scale (float): Scale factor.
        optical_center (np.ndarray or list): Optical center of the image.
        out_res (tuple): Output resolution of the transformed image.
        rot (float, optional): Rotation angle in radians. Default is 0.

    Returns:
        tuple: A tuple containing two transformation matrices:
               - Total transformation matrix.
               - Post rotation transformation matrix.
    """
    rot_mat = np.zeros((3, 3))
    sn, cs = np.sin(rot), np.cos(rot)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1
    # Rotate center to obtain coordinate of center in rotated image
    origin_rot_center = rot_mat.dot(center.tolist() + [1])[:2]
    # Get center for transform with verts rotated around optical axis
    # (through pixel center, smthg like 128, 128 in pixels and 0,0 in 3d world)
    # For this, rotate the center but around center of image (vs 0,0 in pixel space)
    t_mat = np.eye(3)
    t_mat[0, 2] = - optical_center[0]
    t_mat[1, 2] = - optical_center[1]
    t_inv = t_mat.copy()
    t_inv[:2, 2] *= -1
    transformed_center = (
        t_inv.dot(rot_mat).dot(t_mat).dot(center.tolist() + [1])
    )
    post_rot_trans = get_affine_trans_no_rot(origin_rot_center, scale, out_res)
    total_trans = post_rot_trans.dot(rot_mat)
    # check_t = get_affine_transform_bak(center, scale, res, rot)
    # print(total_trans, check_t)
    affinetrans_post_rot = get_affine_trans_no_rot(
        transformed_center[:2], scale, out_res
    )
    return (
        total_trans.astype(np.float32),
        affinetrans_post_rot.astype(np.float32),
    )
    

def get_affine_transform_batch(centers, scales, optical_centers, out_res, rots=None):
    """
    Batch processing of get_affine_transform

    Parameters:
        centers (np.ndarray): [batch_size, 2] array for center points.
        scales (np.ndarray): [batch_size] array for scale factors.
        optical_centers (np.ndarray): [batch_size, 2] array for optical centers.
        out_res (tuple): Output resolution of the transformed image.
        rots (np.ndarray, optional): [batch_size] array for rotation angles in radians. Default is None (0 rotation).

    Returns:
        tuple: A tuple containing two transformation matrices:
               - Total transformation matrix for the entire batch.
               - Post rotation transformation matrix for the entire batch.
    """
    if rots is None:
        rots = np.zeros(centers.shape[0])

    sn, cs = np.sin(rots), np.cos(rots)

    # Create rotation matrices for the entire batch
    rot_mats = np.zeros((centers.shape[0], 3, 3))
    rot_mats[:, 0, :2] = np.stack([cs, -sn], axis=1)
    rot_mats[:, 1, :2] = np.stack([sn, cs], axis=1)
    rot_mats[:, 2, 2] = 1

    origin_rot_centers = rot_mats @ np.hstack([centers, np.ones((centers.shape[0], 1))])[:, :, None]
    origin_rot_centers = origin_rot_centers[:, :2, 0]

    t_mats = np.eye(3)[None].repeat(centers.shape[0], axis=0)
    t_mats[:, 0, 2] = -optical_centers[:, 0]
    t_mats[:, 1, 2] = -optical_centers[:, 1]

    t_inv = t_mats.copy()
    t_inv[:, :2, 2] *= -1

    transformed_centers = t_inv @ rot_mats @ t_mats @ np.hstack([centers, np.ones((centers.shape[0], 1))])[:, :, None]
    transformed_centers = transformed_centers[:, :2, 0]

    # Replace this with your batch version of get_affine_trans_no_rot
    post_rot_trans = get_affine_trans_no_rot_batch(origin_rot_centers, scales, out_res)
    total_trans = post_rot_trans @ rot_mats

    # Replace this with your batch version of get_affine_trans_no_rot
    affinetrans_post_rot = get_affine_trans_no_rot_batch(transformed_centers, scales, out_res)

    return (
        total_trans.astype(np.float32),
        affinetrans_post_rot.astype(np.float32),
    )

def get_affine_transform_test(center, scale, res, rot=0):
    """
    Computes the affine transformation matrix based on the given parameters 
    and a secondary post-rotation transformation. Note that an image is rotated
    around the image center instead of the optical center.
    
    Parameters:
        center (np.ndarray or list): Center point.
        scale (float): Scale factor.
        res (tuple): Target resolution of the transformed image.
        rot (float, optional): Rotation angle in radians. Default is 0.
        
    Returns:
        tuple: 
            - Total transformation matrix as a float32 type numpy array.
            - Post rotation transformation matrix as a float32 type numpy array.
    """
    rot_mat = np.zeros((3, 3))
    sn, cs = np.sin(rot), np.cos(rot)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1
    # Rotate center to obtain coordinate of center in rotated image
    origin_rot_center = rot_mat.dot(center.tolist() + [
        1,
    ])[:2]
    # Get center for transform with verts rotated around optical axis
    # (through pixel center, smthg like 128, 128 in pixels and 0,0 in 3d world)
    # For this, rotate the center but around center of image (vs 0,0 in pixel space)
    t_mat = np.eye(3)
    t_mat[0, 2] = -res[1] / 2
    t_mat[1, 2] = -res[0] / 2
    t_inv = t_mat.copy()
    t_inv[:2, 2] *= -1
    transformed_center = t_inv.dot(rot_mat).dot(t_mat).dot(center.tolist() + [
        1,
    ])
    post_rot_trans = get_affine_trans_no_rot(origin_rot_center, scale, res)
    total_trans = post_rot_trans.dot(rot_mat)
    # check_t = get_affine_transform_bak(center, scale, res, rot)
    # print(total_trans, check_t)
    affinetrans_post_rot = get_affine_trans_no_rot(transformed_center[:2],
                                                   scale, res)
    return total_trans.astype(np.float32), affinetrans_post_rot.astype(
        np.float32)

def get_affine_trans_no_rot(center, scale, res):
    """
    Computes the affine transformation matrix without rotation based on the given parameters.
    
    Parameters:
        center (np.ndarray or list): Center point.
        scale (float): Scale factor.
        res (tuple): Target resolution of the image.
        
    Returns:
        np.ndarray: Affine transformation matrix.
    """
    affinet = np.zeros((3, 3))
    affinet[0, 0] = float(res[1]) / (scale + 1e-8)
    affinet[1, 1] = float(res[0]) / (scale + 1e-8)
    affinet[0, 2] = res[1] * (-float(center[0]) / (scale + 1e-8) + .5)
    affinet[1, 2] = res[0] * (-float(center[1]) / (scale + 1e-8) + .5)
    affinet[2, 2] = 1
    return affinet

def get_affine_trans_no_rot_batch(centers, scales, res):
    """
    Batch processing of get_affine_trans_no_rot
    
    Parameters:
        centers (np.ndarray): [batch_size, 2] array for center points.
        scales (np.ndarray): [batch_size] array for scale factors.
        res (tuple): Target resolution of the image for all images in the batch.
        
    Returns:
        np.ndarray: [batch_size, 3, 3] array of affine transformation matrices.
    """
    batch_size = centers.shape[0]
    affinets = np.zeros((batch_size, 3, 3))
    
    affinets[:, 0, 0] = float(res[1]) / (scales + 1e-8)
    affinets[:, 1, 1] = float(res[0]) / (scales + 1e-8)
    affinets[:, 0, 2] = res[1] * (-centers[:, 0] / (scales + 1e-8) + .5)
    affinets[:, 1, 2] = res[0] * (-centers[:, 1] / (scales + 1e-8) + .5)
    affinets[:, 2, 2] = 1
    
    return affinets

def get_affine_transform_bak(center, scale, res, rot):
    """
    Backup method to compute an affine transformation matrix with an optional rotation.
    
    Parameters:
        center (np.ndarray or list): Center point.
        scale (float): Scale factor.
        res (tuple): Target resolution of the transformed image.
        rot (float): Rotation angle in radians.
        
    Returns:
        tuple:
            - Transformation matrix as a float32 type numpy array.
            - Another transformation matrix (identical to the first).
    """
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / (scale + 1e-8)
    t[1, 1] = float(res[0]) / (scale + 1e-8)
    t[0, 2] = res[1] * (-float(center[0]) / (scale + 1e-8) + .5)
    t[1, 2] = res[0] * (-float(center[1]) / (scale + 1e-8) + .5)
    t[2, 2] = 1
    if rot != 0:
        rot_mat = np.zeros((3, 3))
        sn, cs = np.sin(rot), np.cos(rot)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t))).astype(np.float32)
    return t, t


def gen_cam_param(joint, pts_2d, mode='ortho'):
    """
    Generates camera parameters for a set of 3D joints projected onto 2D key points.
    
    Parameters:
        joint (np.ndarray): 3D joint positions.
        pts_2d (np.ndarray): 2D key points, which are projections of the 3D joints.
        mode (str, optional): Projection mode, either 'ortho' for orthogonal 
            or 'persp' for perspective. Default is 'ortho'.
            
    Returns:
        np.ndarray: Camera parameters.
        
    Raises:
        Exception: If an unknown mode type is provided.
    """
    if mode in ['persp', 'perspective']:
        pts_2d = pts_2d.reshape(-1)[:, np.newaxis]  # (42, 1)
        joint = joint / joint[:, 2:]
        joint = joint[:, :2]
        jM = np.zeros((42, 2), dtype="float32")
        for i in range(joint.shape[0]):  # 21
            jM[2 * i][0] = joint[i][0]
            jM[2 * i + 1][1] = joint[i][1]
        pad2 = np.array(range(42))
        pad2 = (pad2 % 2)[:, np.newaxis]
        pad1 = (1 - pad2)

        jM = np.concatenate([jM, pad1, pad2], axis=1)  # (42, 4)
        jMT = jM.transpose()  # (4, 42)print
        jMTjM = np.matmul(jMT, jM)  # (4,4)
        jMTb = np.matmul(jMT, pts_2d)
        cam_param = np.matmul(np.linalg.inv(jMTjM), jMTb)
        cam_param = cam_param.reshape(-1)
        return cam_param
    elif mode in ['ortho', 'orthogonal']:
        # ortho only when
        assert np.sum(np.abs(joint[0, :])) == 0
        joint = joint[:, :2]  # (21, 2)
        joint = joint.reshape(-1)[:, np.newaxis]
        pts_2d = pts_2d.reshape(-1)[:, np.newaxis]
        pad2 = np.array(range(42))
        pad2 = (pad2 % 2)[:, np.newaxis]
        pad1 = (1 - pad2)
        jM = np.concatenate([joint, pad1, pad2], axis=1)  # (42, 3)
        jMT = jM.transpose()  # (3, 42)
        jMTjM = np.matmul(jMT, jM)
        jMTb = np.matmul(jMT, pts_2d)
        cam_param = np.matmul(np.linalg.inv(jMTjM), jMTb)
        cam_param = cam_param.reshape(-1)
        return cam_param
    else:
        raise Exception("Unkonwn mode type. should in ['persp', 'orth']")
