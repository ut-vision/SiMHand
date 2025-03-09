import os
import math
from pprint import pformat

from easydict import EasyDict as edict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import CometLogger
from src.constants import (
    COMET_KWARGS,
    SIMCLR_CONFIG,
    PECLR_CONFIG,
    SIMHAND_CONFIG,
    SIMHAND_MV_CONFIG,
    BASE_DIR,
    TRAINING_CONFIG_PATH,
)
from src.data_loader.data_set import Data_Set
from src.data_loader.utils import get_data, get_train_val_split
from src.experiments.utils import (
    get_callbacks,
    get_general_args,
    get_model,
    prepare_name,
    save_experiment_key,
    update_model_params,
    update_train_params,
)

from src.utils import get_console_logger, read_json

from pytorch_lightning.loggers import TensorBoardLogger


def main():
    # get configs
    console_logger = get_console_logger(__name__)
    args = get_general_args(f"Model training script.")

    print(f"Model Name: {args.experiment_type}")
    print(f"Dataset Name: {args.sources}")
    print(f"Dataset Scale: {args.datasets_scale}")
    source_datasets_scale = [f"{source}-{args.datasets_scale}" for source in args.sources]
    for source in args.sources:
        if 'freihand' in source or 'youtube' in source:
            source_datasets_scale = ['fh_yt3d']
           
    print(f"Dataset Name-Scale: {source_datasets_scale}")
    print(f"Learning Rate: {args.lr}")

    lr = 1e-4
    lr_str = lr * math.sqrt(1024 * args.accumulate_grad_batches)
    lr_str = f"{lr_str:.1e}"

    print(f"Batch Size: {args.batch_size / 1024} * 1024")
    print(f"Epochs: {args.epochs}")
    print(f"GPUs Use: {args.gpus}")

    if 'w' in args.experiment_type:
        print(f"Weight Type: {args.weight_type}")
        print(f"Joints Type: {args.joints_type}")
        print(f"Joints Difference: {args.diff_type}")
        print(f"Pos Neg: {args.pos_neg}")
        print(f"USE PCA?: {args.use_pca}")
        if args.weight_type == 'non_linear':
            print(f"Non-linear Lambda Pos: {args.non_linear_lambda_pos}")
            print(f"Non-linear Lambda Neg: {args.non_linear_lambda_neg}")
    
    train_param = edict(read_json(TRAINING_CONFIG_PATH))
    train_param = update_train_params(args, train_param)

    if 'simclr' in args.experiment_type:
        model_param_path =  SIMCLR_CONFIG
    elif 'peclr' in args.experiment_type:
        model_param_path =  PECLR_CONFIG
    elif 'simhand' in args.experiment_type:
        model_param_path =  SIMHAND_CONFIG
    else:
        raise ValueError(f"Model {args.experiment_type} is not supported.")
    
    if args.debug is True:
        import logging
        from src.experiments.utils import setup_logging
        logger = setup_logging(log_directory='logs')
        logger.debug('This is a debug message that will be logged to debug.log.')
    else:
        logger = None

    model_param = edict(read_json(model_param_path))
    console_logger.info(f"Train parameters {pformat(train_param)}")
    seed_everything(train_param.seed)

    # data preperation
    data = get_data(
        Data_Set, train_param, sources=args.sources, experiment_type=args.experiment_type, datasets_scale = args.datasets_scale, logger = logger
    )

    train_data_loader, _ = get_train_val_split(
        data, batch_size=train_param.batch_size, num_workers= train_param.num_workers, pin_memory=True, persistent_workers=True
    )
    
    # Logger
    experiment_name = prepare_name(
        f"{args.experiment_type}_", train_param, hybrid_naming=False
    )
    comet_logger = CometLogger(**COMET_KWARGS, experiment_name=experiment_name)

    experiment_key = str(comet_logger.experiment.get_key())
    
    args.vis_save_dir = os.path.join(
        BASE_DIR, "data", "models", f"{experiment_key}" # Directory to save visualization results
    )

    # Initialize TensorBoard Logger
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(
        BASE_DIR, "data", "models", f"{experiment_key}", "tensorboard_logs"   # Directory to save TensorBoard logs
        ),
        name=f'{args.experiment_type}',  # Experiment name or directory name for TensorBoard logs
        version=1  # You can increment this if you want to create new logs for different experiments
    )

    # model
    model_param = update_model_params(model_param, args, len(data), train_param)

    model_param.augmentation = [
        key for key, value in train_param.augmentation_flags.items() if value
    ]
    console_logger.info(f"Model parameters {pformat(model_param)}")
    
    if not args.resume and not args.eval:
        mode = 'train'
    else:
        mode = 'eval'

    model = get_model(
        experiment_type=args.experiment_type,
        heatmap_flag=args.heatmap,
        denoiser_flag=args.denoiser,        
    )(config=model_param, logger_debug = logger, mode = mode)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=args.save_top_k,
        monitor="contrastive_loss",
        mode='min', # Due we moniter the loss value, we use the min mode
        # dirpath="my/path/",
        filename=f"{args.experiment_type}_pretrain_{{epoch:02d}}_train_{source_datasets_scale}_bs_{args.batch_size / 1024}_{1024 * args.accumulate_grad_batches}_lr_{lr_str}_{{contrastive_loss:.6f}}",
    )

    # trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=[0,1,2,3,4,5,6,7],  # Use 8 GPUs
        strategy="dp",
        logger=[comet_logger, tb_logger],  # Include TensorBoardLogger here
        max_epochs=train_param.epochs,
        precision=train_param.precision,
        amp_backend="native",
        replace_sampler_ddp=False, # Cancel DP sampler
        callbacks=[checkpoint_callback],
        log_every_n_steps=5
    )
    
    trainer.logger.experiment.set_code(
        overwrite=True,
        filename=os.path.join(
            BASE_DIR, "src", "experiments", "main_pretrain.py"
        ),
    )
    if args.meta_file is not None:
        save_experiment_key(
            experiment_name, trainer.logger.experiment.get_key(), args.meta_file
        )
    trainer.logger.experiment.log_parameters(train_param)
    trainer.logger.experiment.log_parameters(model_param)
    trainer.logger.experiment.add_tags(["pretraining", f"{args.experiment_type}"] + args.tag)
    
    if not args.resume and not args.eval:
        trainer.fit(model, train_dataloaders = train_data_loader, val_dataloaders=None)
    elif args.resume:
        if args.resume_path is not None:
            message = f"RESUME {args.experiment_type} MODEL TRAINING"
            print(f"""\n\n###################################\t{message.upper()}\t###################################""")
            trainer.fit(model, train_dataloaders = train_data_loader, val_dataloaders=None, ckpt_path="/your/path/to/checkpoint.ckpt")
        else:
            raise ValueError(f"{args.resume_path} is empty, please cheack about it!")

    elif args.eval:
        if args.eval_path is not None:
            message = f"EVAL {args.experiment_type} MODEL TESTING"
            print(f"""\n\n###################################\t{message.upper()}\t###################################""")
            trainer.test(model, dataloaders=train_data_loader, ckpt_path="")
        else:
            raise ValueError(f"{args.resume_path} is empty, please cheack about it!")
    
    print(f"For the {args.experiment_type} model contrastive learning, the best checkpoint is: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()