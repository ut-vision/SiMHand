from typing import Dict, Tuple

import os
import numpy as np

import torch
from easydict import EasyDict as edict
from src.models.unsupervised.simclr_model import SimCLR
from src.models.utils import (
    rotate_encoding,
    translate_encodings,
    translate_encodings2,
    vanila_contrastive_loss,
)
from torch import Tensor
from torch.nn import functional as F


class HandCLR_VIS(SimCLR):
    """
    Implementation Data: 2024.01.19
    HandCLR implementation inspired from method: SimCLR & PeCLR
    SimCLR paper: https://arxiv.org/pdf/2002.05709.pdf.
    PeCLR paper: https://arxiv.org/abs/2106.05953.
    """
    def __init__(self, config: edict, logger_debug: None, mode: str):
        super().__init__(config, logger_debug, mode)
        
        self.save_hyperparameters()

        self.iteration = 0  # iteration number
        self.save_dir = 'PATH/TO/SAVE/DIR'  # path to save the transformed projections
    
    def get_transformed_projections(
        self, batch: Dict[str, Tensor], iteration: int
    ) -> Tuple[Tensor, Tensor]:
        batch_ori = torch.cat(
            (batch["image1"], batch["image2"]), dim=0
        )

        batch_transform = torch.cat(
            (batch["transformed_image1"], batch["transformed_image2"]), dim=0
        )

        batch_joints_ori = torch.cat(
            (batch["joints1_ori"], batch["joints2_ori"]), dim=0
        )

        batch_joints_aug = torch.cat(
            (batch["joints1_aug"], batch["joints2_aug"]), dim=0
        )

        os.makedirs(self.save_dir, exist_ok=True)

        data = {
            "batch_ori": batch_ori.cpu().numpy(),
            "batch_transform": batch_transform.cpu().numpy(),
            "batch_joints_ori": batch_joints_ori.cpu().numpy(),
            "batch_joints_aug": batch_joints_aug.cpu().numpy()
        }

        filename = f"{self.save_dir}/iteration_{iteration}.npy"

        np.save(filename, data)

        # Assuming shape of batch is [batch, channels, width, height]
        image1_shape = batch["transformed_image1"].size()[-2:]
        image2_shape = batch["transformed_image2"].size()[-2:]
        batch_size = int(len(batch_transform) / 2)
        encodings = self.encoder(batch_transform)

        if self.logger_debug is not None:
            ###########################################################
            # Note: You can use it for debugging purposes
            ###########################################################
            print(f"################# DEBUG MODEL #####################")
        
        projections = self.projection_head(encodings).view((batch_size * 2, -1, 2))

        projection1_stat = self.get_projection_stats(
            projections[:batch_size].detach(), "proj1"
        )
        projection2_stat = self.get_projection_stats(
            projections[batch_size:].detach(), "proj2"
        )
        # normalizing before rotation
        projections = projections.view((batch_size * 2, -1))
        norm_projection1 = F.normalize(projections[:batch_size])
        norm_projection2 = F.normalize(projections[batch_size:])
        projections = torch.cat([norm_projection1, norm_projection2], dim=0).view(
            (batch_size * 2, -1, 2)
        )
        self.train_metrics = {
            **self.train_metrics,
            **projection1_stat,
            **projection2_stat,
        }
        if "crop" in self.config.augmentation:
            jitter_x = torch.cat(
                (
                    batch["jitter_x_1"] / float(image1_shape[0]),
                    batch["jitter_x_2"] / float(image2_shape[0]),
                ),
                dim=0,
            )
            jitter_y = torch.cat(
                (
                    batch["jitter_y_1"] / float(image1_shape[1]),
                    batch["jitter_y_2"] / float(image2_shape[1]),
                ),
                dim=0,
            )
            # moving the encodings by same amount.
            projections = translate_encodings(projections, -jitter_x, -jitter_y, self.logger_debug)

        if "rotate" in self.config.augmentation:
            angles = torch.cat((batch["angle_1"], batch["angle_2"]), dim=0)
            # rotating the projections in opposite direction
            projections = rotate_encoding(projections, -angles, self.logger_debug)

        projections = projections.view((batch_size * 2, -1))
        projection1 = F.normalize(projections[:batch_size])
        projection2 = F.normalize(projections[batch_size:])
        return projection1, projection2

    def contrastive_step(self, batch: Dict[str, Tensor]) -> Tensor:
        self.iteration += 1

        projection1, projection2 = self.get_transformed_projections(batch, self.iteration)
        loss = vanila_contrastive_loss(projection1, projection2)

        self.log("contrastive_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def get_projection_stats(self, projection: Tensor, name: str) -> dict:
        projection_mean = torch.mean(projection, dim=1)
        projection_median = torch.median(projection, dim=1).values
        projection_min = torch.min(projection, dim=1).values
        projection_max = torch.max(projection, dim=1).values
        return {
            f"{name}x_mean": torch.mean(projection_mean, dim=0)[0],
            f"{name}x_median": torch.mean(projection_median, dim=0)[0],
            f"{name}x_min": torch.mean(projection_min, dim=0)[0],
            f"{name}x_max": torch.mean(projection_max, dim=0)[0],
            f"{name}y_mean": torch.mean(projection_mean, dim=0)[1],
            f"{name}y_median": torch.mean(projection_median, dim=0)[1],
            f"{name}y_min": torch.mean(projection_min, dim=0)[1],
            f"{name}y_max": torch.mean(projection_max, dim=0)[1],
        }