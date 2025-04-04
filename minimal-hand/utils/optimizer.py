import torch.optim as optim

def config_optimizer(model, pattern, strategy, lr, logger):
    """
    Configures the optimizer with different learning rate strategies for a finetune model.

    Parameters:
    - model: The model for which the optimizer is being configured.
    - strategy: A string indicating the learning rate strategy. 
      Options include 'ULR' (Uniform Small Learning Rate), 'DLR' (Differential Learning Rates), 
      'GU' (Gradual Unfreezing), 'CLR' (Cyclical Learning Rates), and 'LRW' (Learning Rate Warmup).
    - learning_rates: A dictionary containing learning rates and possibly other parameters 
      relevant to the chosen strategy.

    Returns:
    - Configured optimizer.
    """
    optimizer = None
    
    # Config1: # Uniform Learning Rate (Recommod Small Learning Rate)
    
    if pattern == 'finetune':
        if strategy == 'ULR':  
            optimizer = optim.Adam([
                    {
                        'params': model.parameters(),
                        'initial_lr': float(lr[0])
                    },
                ],
                lr = float(lr[0])
            )
        # Differential Learning Rates (DLR)
        elif strategy == 'DLR':  # Differential Learning Rates  
            # 初始化两个参数列表
            encoder_params = []
            other_params = []

            # 遍历模型的所有参数
            for name, param in model.named_parameters():
                if "encoder" in name:
                    # 如果参数名中包含 "encoder"，则认为它属于 encoder 部分
                    encoder_params.append(param)
                else:
                    # 否则，认为它属于模型的其他部分
                    other_params.append(param)

            # 使用这两组参数来创建优化器，分别设置不同的学习率
            optimizer = optim.Adam([
                {'params': encoder_params, 'lr': float(lr[0])},  # 预训练部分设置学习率
                {'params': other_params, 'lr': float(lr[1])},  # 除了encoder之外的其他部分设置学习率
            ])

            # 首先，我们计算两个分组参数列表的总长度
            total_grouped_params = len(encoder_params) + len(other_params)

            # 然后，我们获取模型所有参数的总长度
            total_model_params = len(list(model.module.parameters()))

            # 使用 assert 语句来确保这两个总数是相等的
            assert total_grouped_params == total_model_params, \
                f"The total number of optimize layers in the model does not match. Please check the model optimize: Model has {total_model_params} parameters, but grouped {total_grouped_params}."

            # 如果没有触发 AssertionError，说明分组正确
            print(f"Parameter grouping check passed. Total number of optimize layers is {total_model_params}")

            '''
            #@ 2024.03.31 廢除V1的DLR優化器版本
            # 设置不同的学习率
            optimizer = optim.Adam([
                {'params': model.module.encoder.parameters(), 'lr': float(lr[0])},  #  预训练部分设置学习率
                # @2024.03.16 Heatmap-base checkpoint load, cancel the [adapter] training
                # {'params': model.module.adapter.parameters(), 'lr': float(lr[1])},  # 适配器层部分设置学习率
                {'params': model.module.hmap_0.parameters(), 'lr': float(lr[1])},  # 热图层部分设置学习率
                {'params': model.module.dmap_0.parameters(), 'lr': float(lr[1])},  # 深度图层部分设置学习率
                {'params': model.module.lmap_0.parameters(), 'lr': float(lr[1])},  # 标签图层部分设置学习率
            ])

            assert len(list(model.module.parameters())) == len(list(model.module.encoder.parameters())) \
                  + len(list(model.module.hmap_0.parameters())) + len(list(model.module.dmap_0.parameters())) + len(list(model.module.lmap_0.parameters())), \
                    f"The total number of optimize layers in the model does not match. Please check the model optimize: {len(model.module.parameters())}."
            '''

            # @2024.03.16 Heatmap-base checkpoint load, cancel the [adapter] training
            # assert len(list(model.module.parameters())) == len(list(model.module.resnet50.parameters())) \
            #       + len(list(model.module.adapter.parameters())) + len(list(model.module.hmap_0.parameters())) \
            #       + len(list(model.module.dmap_0.parameters())) + len(list(model.module.lmap_0.parameters())), \
            #         f"The total number of optimize layers in the model does not match. Please check the model optimize: {len(model.module.parameters())}."

        elif strategy == 'GU':  # Gradual Unfreezing
            # This strategy requires implementation in the training loop rather than a simple optimizer configuration.
            pass

        elif strategy == 'CLR':  # Cyclical Learning Rates or Learning Rate Warm Restarts
            # This strategy typically requires a scheduler to adjust the learning rate cyclically, not covered here.
            pass

        elif strategy == 'LRW':  # Learning Rate Warmup
            # This strategy typically requires a scheduler to gradually increase the learning rate, not covered here.
            pass

        else:
            raise ValueError("Unsupported learning rate strategy")
    
    else:
        optimizer = optim.Adam([
                    {
                        'params': model.parameters(),
                        'initial_lr': float(lr[0])
                    },
                ],
                lr = float(lr[0])
            )

    return optimizer
