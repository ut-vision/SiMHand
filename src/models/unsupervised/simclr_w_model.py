from typing import Dict, Tuple

from typing import Dict

import torch
from easydict import EasyDict as edict
from src.models.base_model import BaseModel
from src.models.utils import vanila_contrastive_loss
from src.models.utils import (
    rotate_encoding,
    translate_encodings,
    translate_encodings2,
    vanila_weights_contrastive_loss,
    vanila_pos_weights_contrastive_loss,
    vanila_neg_weights_contrastive_loss,
    get_weights_linear,
    get_weights_nonlinear,
)
from torch import Tensor, nn


class SimCLR_W(BaseModel):
    """
    SimcLR implementation inspired from paper https://arxiv.org/pdf/2002.05709.pdf.
    The code is adapted from pl_bolts library.
    """
    def __init__(self, config: edict, logger_debug: None, mode: str):
        super().__init__(config, logger_debug, mode)
        self.save_hyperparameters()
        self.projection_head = self.get_projection_head()
        self.mode = mode

    def get_projection_head(self) -> nn.Sequential:
   
        projection_head = nn.Sequential(
            nn.Linear(
                self.config.projection_head_input_dim,
                self.config.projection_head_hidden_dim,
                bias=True,
            ),
            nn.BatchNorm1d(self.config.projection_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.config.projection_head_hidden_dim,
                self.config.output_dim,
                bias=False,
            ),
        )

        return projection_head

    def get_adaptive_weights(
        self, batch: Dict[str, Tensor], joints_type, weight_type, diff_type
    ) -> Tuple[Tensor, Tensor]:
        
        if joints_type is 'original':
            joints1 = batch["joints1_ori"][:, :, :2]
            joints2 = batch["joints2_ori"][:, :, :2]
        else:
            joints1 = batch["joints1_aug"][:, :, :2]
            joints2 = batch["joints2_aug"][:, :, :2]

        if weight_type == 'linear':
            pos_weights, neg_weights = get_weights_linear(joints1, joints2, diff_type)
        elif weight_type == 'non_linear':
            pos_weights, neg_weights = get_weights_nonlinear(joints1, joints2, self.config.non_linear_lambda_pos, self.config.non_linear_lambda_neg, diff_type)

        return pos_weights, neg_weights


    def contrastive_step(self, batch: Dict[str, Tensor]) -> Tensor:

        batch_size = batch["transformed_image1"].size()[0]
        concat_batch = torch.cat(
            (batch["transformed_image1"], batch["transformed_image2"]), dim=0
        )
        
        concat_encoding = self.get_encodings(concat_batch)
        concat_projections = self.projection_head(concat_encoding)
        projection1, projection2 = (
            nn.functional.normalize(concat_projections[:batch_size]),
            nn.functional.normalize(concat_projections[batch_size:]),
        )
        
        pos_weights, neg_weights = self.get_adaptive_weights(batch, self.config.joints_type, self.config.weight_type, self.config.diff_type)
        
        if self.config.pos_neg == "pos_neg":
            loss = vanila_weights_contrastive_loss(projection1, projection2, pos_weights, neg_weights)
        elif self.config.pos_neg == "pos":
            loss = vanila_pos_weights_contrastive_loss(projection1, projection2, pos_weights)
        elif self.config.pos_neg == "neg":
            loss = vanila_neg_weights_contrastive_loss(projection1, projection2, neg_weights)
        
        self.log("contrastive_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def get_encodings(self, batch_images: Tensor) -> Tensor:

        return self.encoder(batch_images)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:

        embedding = self.encoder(x)
        projection = self.projection_head(embedding)
        return {"embedding": self.encoder(x), "projection": projection}

    def training_step(self, batch: dict, batch_idx: int) -> Dict[str, Tensor]:
    
        loss = self.contrastive_step(batch)
        self.train_metrics = {**self.train_metrics, **{"loss": loss}}
        self.plot_params = {
            "image1": batch["transformed_image1"],
            "image2": batch["transformed_image2"],
            "params": {k: v for k, v in batch.items() if "image" not in k},
        }
        return self.train_metrics

    def validation_step(self, batch: dict, batch_idx: int) -> Dict[str, Tensor]:
        loss = self.contrastive_step(batch)
        self.plot_params = {
            "image1": batch["transformed_image1"],
            "image2": batch["transformed_image2"],
            "params": {k: v for k, v in batch.items() if "image" not in k},
        }
        return {"loss": loss}