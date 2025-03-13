import torch
from torchvision.models import resnet50

dependencies = ['torch', 'torchvision']  # 声明依赖项

def resnet50_simhand(pretrained=False, version='v1.0', **kwargs):
    """ 
    加载使用手部数据预训练的ResNet50模型
    Args:
        pretrained (bool): 是否加载预训练权重
    """
    model = resnet50(pretrained=False)
    
    if pretrained:
        # 从GitHub Releases加载权重
        checkpoint_url = f"https://github.com/ut-vision/SiMHand/releases/download/{version}/resnet50_simhand.pth"
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint_url,
            map_location=torch.device('cpu'),
            progress=True
        )
        model.load_state_dict(state_dict)
        
    return model