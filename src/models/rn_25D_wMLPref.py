import torch
import torch.nn as nn
from torchvision import models


class ZrootMLP_ref(nn.Module):
    """
    Zroot refinement module taken from: https://arxiv.org/abs/2003.09282
    Given 21 2D and zrel keypoints, plus a zroot estimate, refines the zroot estimate
    via:
    zroot_ref = zroot_est + mlp(2D, zrel, zroot_est)
    """

    def __init__(self):
        super().__init__()

        zroot_ref = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )
        self.norm_bone_idx = (3, 8)
        self.zroot_ref = zroot_ref
        epsilon = 1e-8
        self.register_buffer("eps", torch.tensor(epsilon), persistent=False)

    def forward(self, kp3d_unnorm, zrel, K):
        eps = self.eps
        # Recover the scale normalized zroot using https://arxiv.org/pdf/1804.09534.pdf
        m = self.norm_bone_idx[0]
        n = self.norm_bone_idx[1]
        X_m = kp3d_unnorm[:, m : m + 1, 0:1]
        Y_m = kp3d_unnorm[:, m : m + 1, 1:2]
        X_n = kp3d_unnorm[:, n : n + 1, 0:1]
        Y_n = kp3d_unnorm[:, n : n + 1, 1:2]
        zrel_m = zrel[:, m : m + 1]
        zrel_n = zrel[:, n : n + 1]
        # Eq (6)
        a = (X_n - X_m) ** 2 + (Y_n - Y_m) ** 2
        b = 2 * (
            zrel_n * (X_n ** 2 + Y_n ** 2 - X_n * X_m - Y_n * Y_m)
            + zrel_m * (X_m ** 2 + Y_m ** 2 - X_n * X_m - Y_n * Y_m)
        )
        c = (
            (X_n * zrel_n - X_m * zrel_m) ** 2
            + (Y_n * zrel_n - Y_m * zrel_m) ** 2
            + (zrel_n - zrel_m) ** 2
            - 1
        )
        d = (b ** 2) - (4 * a * c)
        # Push sufficiently far away from zero to ensure numerical stability
        a = torch.max(eps, a)
        d = torch.max(eps, d)
        # Eq (7)
        zroot = ((-b + torch.sqrt(d)) / (2 * a)).detach()
        # Refine zroot estimate via an MLP using: https://arxiv.org/abs/2003.09282
        zroot = torch.clamp(zroot, 4.0, 50.0)
        mlp_input = torch.cat(
            (
                zrel.reshape(-1, 21),
                kp3d_unnorm[..., :2].reshape(-1, 42),
                zroot.reshape(-1, 1),
            ),
            dim=1,
        )
        zroot = zroot + self.zroot_ref(mlp_input).reshape(zroot.shape)

        return zroot


class RN_25D_wMLPref(nn.Module):
    def __init__(self, backend_model="rn50"):
        super().__init__()
        # Initialize a torchvision resnet
        if backend_model == "rn50":
            model_func = models.resnet50
        elif backend_model == "rn152":
            model_func = models.resnet152
        else:
            raise Exception(f"Unknown backend_model: {backend_model}")
        backend_model = model_func()
        num_feat = backend_model.fc.in_features
        # 2D + zrel for 21 keypoints: 3 * 21. Please ignore +1, it is no longer used
        backend_model.fc = nn.Linear(num_feat, 3 * 21 + 1)
        # Initialize the zroot refinement module
        zroot_ref = ZrootMLP_ref()

        self.backend_model = backend_model
        self.zroot_ref = zroot_ref
        self.register_buffer(
            "K_default",
            torch.Tensor(
                [
                    [388.9018310596544, 0.0, 112.0],
                    [0.0, 388.71231836584275, 112.0],
                    [0.0, 0.0, 1.0],
                ]
            ).reshape(1,3,3),
            persistent=False,
        )

    def forward(self, img, K=None):
        '''
        # ----------------------- visualization ----------------------------- #
        import time
        import cv2
        import os
        import numpy as np
        from src.keypoints_vis.vis import draw_2d_skeleton
        image_mean = np.array([0.485, 0.456, 0.406])
        image_std = np.array([0.229, 0.224, 0.225])
        print(f"圖像img的形狀是{img.shape}")
        # 将张量从GPU移到CPU并转换成NumPy数组，再将其从[1, 3, 224, 224]转换成[224, 224, 3]
        img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # 反标准化图片数据
        img_np = (img_np * image_std + image_mean)

        # 确保数据在0到1之间
        img_np = np.clip(img_np, 0, 1)

        # 转换成255的范围并转换成uint8类型
        img_np = (img_np * 255).astype(np.uint8)

        # 如果原始图片是RGB格式，转换成BGR格式用于保存
        img_np = img_np[:, :, [2, 1, 0]]

        save_dir = '/large/nielin/code/handclr/visualization/rn_25D_wMLPref'
        # ret_2d = draw_2d_skeleton(img, joints25D[:, :2])
        # img_id = sample["image_name"].split('/')[-1]
        img_id = '00000000'
        print(os.path.join(save_dir, img_id + '_after_normal.png'))
        # cv2.imwrite(os.path.join(save_dir, img_id + '_after_normal.png'), img_np)
        # time.sleep(10)
        # ----------------------- visualization ----------------------------- #
        
        # ---------------- 查看K的值 --------------------- #
        print(f"K的值是{K}")
        # time.sleep(10)
        # ---------------- 查看K的值 --------------------- #
        '''
        if K is None:
            # Use a default camera matrix
            K = self.K_default

        out = self.backend_model(img)

        # ---------------- 查看out的值 --------------------- #
        # print(f"out的值是{out}")
        # print(f"out的形狀是{out.shape}")
        # time.sleep(10)
        # ---------------- 查看out的值 --------------------- #

        kp25d = out[:, :-1].view(-1,21,3)

        kp2d = kp25d[..., :2]
        zrel = kp25d[..., 2:3]
        # ---------------- 查看out的值 --------------------- #
        import time
        print(f"zrel的值是{zrel}")
        print(f"zrel的形狀是{zrel.shape}")
        time.sleep(10)
        # ---------------- 查看out的值 --------------------- #

        # We know that zrel of root is 0
        zrel[:, 0] = 0
        
        # ---------------- 查看out的值 --------------------- #
        import time
        print(f"zrel的值是{zrel}")
        print(f"zrel的形狀是{zrel.shape}")
        time.sleep(10)
        # ---------------- 查看out的值 --------------------- #
        
        # ---------------- 查看kp25d的值 --------------------- #
        # print(f"kp2d的值是{kp2d}")
        # print(f"kp2d的形狀是{kp2d.shape}")


        # print(f"zrel的值是{zrel}")
        # print(f"zrel的形狀是{zrel.shape}")

        # time.sleep(10)
        # ---------------- 查看kp25d的值 --------------------- #

        ''''
        # ----------------------- visualization ----------------------------- #
        import time
        import cv2
        import os
        import numpy as np
        from src.keypoints_vis.vis import draw_2d_skeleton
        print(f"img_np的形狀是{img_np.shape}")
        img_np = img_np.astype(np.uint8)
        
        kp2d_np = kp2d.squeeze(0).cpu().numpy()


        # print(kp2d_np)
        ret_2d = draw_2d_skeleton(img_np, kp2d_np)
        print(os.path.join(save_dir, img_id + '_2d_vis.png'))
        cv2.imwrite(os.path.join(save_dir, img_id + '_2d_vis.png'), ret_2d)
        # time.sleep(10)
        # ----------------------- visualization ----------------------------- #
        '''
        
        # Acquire refined zroot
        kp2d_h = torch.cat(
            (kp2d, torch.ones((kp2d.shape[0], 21, 1), device=K.device)), dim=2
        )   # 为2D坐标kp2d添加一个维度，使其成为齐次坐标，以便进行后续的矩阵运算。
        kp3d_unnorm = torch.matmul(kp2d_h, K.inverse().transpose(1, 2))
        zroot = self.zroot_ref(kp3d_unnorm, zrel, K)

        # ---------------- 查看out的值 --------------------- #
        import time
        print(f"zroot的值是{zroot}")
        print(f"zroot的形狀是{zroot.shape}")
        time.sleep(10)
        # ---------------- 查看out的值 --------------------- #

        # Compute the scale-normalized 3D keypoints using
        # https://arxiv.org/pdf/1804.09534.pdf
        kp3d = kp3d_unnorm * (zrel + zroot)


        # ---------------- 查看kp3d的值 --------------------- #
        import time
        print(f"kp3d的值是{kp3d}")
        print(f"kp3d的形狀是{kp3d.shape}")
        time.sleep(10)
        # ---------------- 查看kp3d的值 --------------------- #
        

        output = {}
        output["kp3d"] = kp3d
        output["zrel"] = zrel
        output["kp2d"] = kp2d
        output['kp25d'] = kp25d
        
        return output
