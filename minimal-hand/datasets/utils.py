import torch
import numpy as np
import matplotlib.pyplot as plt

#from pytorch3d.ops import corresponding_points_alignment
'''
from manopth.manolayer import ManoLayer
from manopth import demo
'''
from datasets import MIDDLE_MCP_ID, REF_BONE_LINK, MANO_MODEL_PATH

class Cam2ManoTransform(object):
    """
    transform camera coordinate to mano coordinate
    - keypoint order follows the mano model
    - root relative and scale normalized joints are projected to align with canonical mano joints    
    """
    def __init__(self, mano_model_path=None, root_idx=None, bone_link=None):
        self.root_idx = root_idx if root_idx is not None else MIDDLE_MCP_ID
        self.bone_link = bone_link if bone_link is not None else REF_BONE_LINK
        mano_model_path = mano_model_path if mano_model_path is not None else MANO_MODEL_PATH
        self.palm_joint_idx = [0, 5, 9, 13, 17]
        ncomps = 45
        mano_layer = ManoLayer(mano_root=mano_model_path, use_pca=True, ncomps=ncomps)
        torch.random.manual_seed(42)
        pose_init = torch.rand(1, ncomps + 3)
        shape_init = torch.rand(1, 10)
        _, self.canonical_joints = mano_layer(pose_init, shape_init)
        self.canonical_joints_RS = self._relative_scale_norm(self.canonical_joints)
        
    def _relative_scale_norm(self, joints, ref_joints=None):
        ref_joints = joints if ref_joints is None else ref_joints
        # root relative 
        rel_joints = joints - joints[:, self.root_idx:self.root_idx+1]
        # scale norm
        joint_bone = torch.norm(ref_joints[:, self.bone_link[0]] - ref_joints[:, self.bone_link[1]], dim=-1)
        rel_norm_joints = rel_joints / joint_bone[:, None, None]
        return rel_norm_joints
    
    def _inv_relative_scale_norm(self, joints_RS, ref_joints, root_joint=None):
        """
        joints_RS: root relative and scale normalized joints
        ref_joints: reference joints for scale normalization (usually GT is given)
        root_joint: root joint for root relative transformation
        """
        root_joint = ref_joints[:, self.root_idx:self.root_idx+1] if root_joint is None else root_joint
        joint_bone = torch.norm(ref_joints[:, self.bone_link[0]] - ref_joints[:, self.bone_link[1]], dim=-1)        
        rel_joints = joints_RS * joint_bone[:, None, None]
        recovered_joints = rel_joints + root_joint
        return recovered_joints
        
    def transform(self, joints, ref_joints=None):
        """palm (rigid) joints are used to compute similarity transform"""
        assert joints.shape[1:] == (21, 3), f"shape error: {joints.shape}"
        joints_RS = self._relative_scale_norm(joints, ref_joints)
        canonical_joints_RS = self.canonical_joints_RS.repeat(joints_RS.shape[0], 1, 1)
        ST = corresponding_points_alignment(joints_RS[:, self.palm_joint_idx], canonical_joints_RS[:, self.palm_joint_idx], estimate_scale=False, allow_reflection=False)
        transformed_joints_RS = torch.bmm(joints_RS, ST.R) + ST.T[:, None, :]
        RT = torch.cat([ST.R, ST.T[:, :, None]], dim=-1) # R (N, 3, 3), T (N, 1, 3) -> RT (N, 3, 4)
        return transformed_joints_RS, RT

    def transform_by_RT(self, joints, RT, ref_joints=None):
        if RT.shape == (3, 4) or RT.shape[0] == 1: # extend to batch
            _RT = RT.view(1, 3, 4).repeat(joints.shape[0], 1, 1)
        assert joints.shape[1:] == (21, 3), f"shape error: {joints.shape}"
        joint_RS = self._relative_scale_norm(joints, ref_joints)
        # assert torch.allclose(torch.norm(joint_RS[:, self.bone_link[0]] - joint_RS[:, self.bone_link[1]], dim=-1), torch.ones_like(joint_RS[:, 0, 0])), f"bone error: {torch.norm(joint_RS[:, self.bone_link[0]] - joint_RS[:, self.bone_link[1]], dim=-1)}"
        transformed_joints_RS = torch.bmm(joint_RS, _RT[:, :, :3]) + _RT[:, :, 3].view(-1, 1, 3)
        # assert torch.allclose(torch.norm(transformed_joints_RS[:, self.bone_link[0]] - transformed_joints_RS[:, self.bone_link[1]], dim=-1), torch.ones_like(transformed_joints_RS[:, 0, 0])), f"bone error: {torch.norm(transformed_joints_RS[:, self.bone_link[0]] - transformed_joints_RS[:, self.bone_link[1]], dim=-1)}"
        return transformed_joints_RS
    
    def inv_transform(self, transformed_joints_RS, RT):
        assert transformed_joints_RS.shape[1:] == (21, 3), f"shape error: {transformed_joints_RS.shape}"
        assert RT.shape[1:] == (3, 4), f"shape error: {RT.shape}"
        R = RT[:, :, :3]
        T = RT[:, :, 3]
        joints_RS = torch.bmm(transformed_joints_RS - T[:, None, :], torch.inverse(R))
        return joints_RS
    
    def compute_velocity(self, RT):
        vel = torch.zeros(RT.shape[0], 4, 4)
        vel[0] = torch.eye(4, 4)
        # homogeneous transformation matrix 
        M = torch.cat([RT, torch.tensor([[[0,0,0,1]]], dtype=torch.float32).repeat(RT.shape[0], 1, 1)], dim=1)
        # vel is computed by the composition of rigid transformations: M (t) * M (t-1)^(-1)
        vel[1:] = torch.bmm(M[1:], torch.inverse(M[:-1]))
        return vel[:, :3]
    
    def compute_project_error(self, transformed_joints_RS, original_joints, root_joint=None):
        assert transformed_joints_RS.shape[1:] == original_joints.shape[1:] == (21, 3), f"shape error: {transformed_joints_RS.shape}, {original_joints.shape}"
        transformed_joints = self._inv_relative_scale_norm(transformed_joints_RS, original_joints, root_joint)
        rel_joints = transformed_joints - transformed_joints[:, root_idx:root_idx+1]
        rel_mano_joints = self.canonical_joints - self.canonical_joints[:, root_idx:root_idx+1]
        rel_mano_joints = rel_mano_joints.repeat(rel_joints.shape[0], 1, 1)
        jpe = torch.norm(rel_joints - rel_mano_joints, dim=-1)
        mpjpe_palm = jpe[:, self.palm_joint_idx].mean(dim=-1).tolist()
        mpjpe_full = jpe.mean(dim=-1).tolist()
        return f"rel error: {mpjpe_full} (full), {mpjpe_palm} (palm)"

def vis_kpt_3d(clr, pts_2d, joint, save_path=None):
    # fig = plt.figure(figsize=(20, 20))
    fig = plt.figure(figsize=(12, 5))
    clr_ = np.array(clr)

    plt.subplot(1, 3, 1)
    clr1 = clr_.copy()
    plt.imshow(clr1)

    plt.subplot(1, 3, 2)
    clr2 = clr_.copy()
    plt.imshow(clr2)

    for p in range(pts_2d.shape[0]):
        plt.plot(pts_2d[p][0], pts_2d[p][1], 'r.')
        plt.text(pts_2d[p][0], pts_2d[p][1], '{0}'.format(p), fontsize=5)

    ax = fig.add_subplot(133, projection='3d')
    plt.plot(joint[:, 0], joint[:, 1], joint[:, 2], 'yo', label='keypoint')
    plt.plot(joint[:5, 0], joint[:5, 1],
                joint[:5, 2],
                'r',
                label='thumb')
    plt.plot(joint[[0, 5, 6, 7, 8, ], 0], joint[[0, 5, 6, 7, 8, ], 1],
                joint[[0, 5, 6, 7, 8, ], 2],
                'b',
                label='index')
    plt.plot(joint[[0, 9, 10, 11, 12, ], 0], joint[[0, 9, 10, 11, 12], 1],
                joint[[0, 9, 10, 11, 12], 2],
                'b',
                label='middle')
    plt.plot(joint[[0, 13, 14, 15, 16], 0], joint[[0, 13, 14, 15, 16], 1],
                joint[[0, 13, 14, 15, 16], 2],
                'b',
                label='ring')
    plt.plot(joint[[0, 17, 18, 19, 20], 0], joint[[0, 17, 18, 19, 20], 1],
                joint[[0, 17, 18, 19, 20], 2],
                'b',
                label='pinky')
    # snap convention
    plt.plot(joint[4][0], joint[4][1], joint[4][2], 'rD', label='thumb')
    plt.plot(joint[8][0], joint[8][1], joint[8][2], 'ro', label='index')
    plt.plot(joint[12][0], joint[12][1], joint[12][2], 'ro', label='middle')
    plt.plot(joint[16][0], joint[16][1], joint[16][2], 'ro', label='ring')
    plt.plot(joint[20][0], joint[20][1], joint[20][2], 'ro', label='pinky')

    plt.title('3D annotations')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend()
    ax.view_init(-90, -90)
    # ax.view_init(-45, -45)
    # plt.show()
    plt.tight_layout()
    save_path = save_path if save_path is not None else "vis_gt.jpg"
    plt.savefig(save_path)
