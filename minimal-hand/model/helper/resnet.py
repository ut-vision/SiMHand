import torch
import torch.nn as nn
import torchvision.models as models


class ResNetModel(nn.Module):
    """Adapted ResNet model with layers renamed.
    """
    # -------------------- Modify 59: Pass the params self.loggger ---------------------- #
    def __init__(self, name, logger, pretrain: None):
        super(ResNetModel, self).__init__()
        self.logger = logger
    # -------------------- Modify 59: Pass the params self.loggger ---------------------- #
        # self.mode = mode
        resnet_name = name
        model_function = self.get_resnet(resnet_name)
        # @2024.04.01 Got Warning: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and will be removed in 0.15, please use 'weights' instead.
        # model = model_function(pretrained=True, norm_layer =nn.BatchNorm2d)
        
        # DONT USE PT BACKBONE
        if pretrain is None:
            self.logger.info(f"Since 'PRETRAIN' set to {pretrain}, no PT weights will be used for model's backbone -- {resnet_name} initialization.")
            model = model_function(norm_layer=nn.BatchNorm2d)
        # USE OFFICIAL PT BACKBONE
        elif 'ResNet' in pretrain:
            self.logger.info(f"With 'PRETRAIN' set to {pretrain}, the model's backbone -- {resnet_name} will be initialized via the latest official IMAGENET PT weights.")
            # weights='ResNet50_Weights.DEFAULT'
            # weights=ResNet50_Weights.IMAGENET1K_V1
            model = model_function(weights=pretrain, norm_layer=nn.BatchNorm2d)  # Change pretrained to weights='imagenet'
        
        '''
        else:
            self.logger.info(f"With 'PRETRAIN' set to Local Pretrained Model Path {pretrain}, the model's backbone -- {resnet_name} will be initialized via the unofficial local pretrained model weights.")
            model = model_function(norm_layer=nn.BatchNorm2d)
            # Load weights if path is provided
            pretrain_weight = torch.load(pretrain, map_location=torch.device('cpu'))  # Load on CPU
            if 'state_dict' in pretrain_weight:
                pretrain_weight = pretrain_weight['state_dict']
            model.load_state_dict(pretrain_weight, strict=False)
            print(f"Local Pretrained Model {pretrain} Weights loaded successfully!!")
        '''

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            # nn.AdaptiveAvgPool2d(output_size=(1,1)),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.final_layer = nn.Sequential(
        #             nn.Linear(model.fc.in_features,21*3+1),
        #             )
    
    def get_resnet(self, resnet_name):
        if "resnet18" == resnet_name:
            return models.resnet18
        elif "resnet34" == resnet_name:
            return models.resnet34
        elif "resnet50" == resnet_name:
            return models.resnet50
        elif "resnet101" == resnet_name:
            return models.resnet101
        elif "resnet152" == resnet_name:
            return models.resnet152
        else:
            raise NotImplementedError
    
    def forward(self, x):
        """Forward method, return embeddings when the mode is pretraining.
        and return 2.5D keypoints, None and scale otherwise.
        """
        # self.logger.info(self.features)
        
        z = self.features(x)

        # elf.logger.info("判断是否有调用features层")
        # self.logger.info(f"特征z的形状是{z.shape}")
        
        # 新增平均池化层，避免特征损失。 
        # z = self.avgpool(z)

        # z = z.flatten(start_dim=1)
        # self.logger.info(f"特征z的形状是{z.shape}")
        # if self.mode=="pretraining":
        
        return z
        
        # else:
        #     z = self.final_layer(z)
        #     return z[:,:21*3], None, z[:,-1]

if __name__ == '__main__':    
    # Example of usage
    weight_path = '/large/nielin/code/handclr/data/models/peclr_rn50_yt3dh_fh/checkpoints/peclr_rn50_yt3dh_fh.pth'
    # 不加载任何模型权重
    model = ResNetModel(name='resnet50', logger=None, pretrain=None)
    # 加载imagenet模型预训练模型权重
    model = ResNetModel(name='resnet50', logger=None, pretrain=None)
