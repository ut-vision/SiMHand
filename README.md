# SiMHand
This is the official implementation of our ICLR 2025 paper "[SiMHand: Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training](https://openreview.net/forum?id=96jZFqM5E0)". Hope to share our work in Singapore 🇸🇬 ～!!

![labram](simhand.png)
## Abstract
We present a framework for pre-training of 3D hand pose estimation from in-the-wild hand images sharing with similar hand characteristics, dubbed SiMHand. Pre-training with large-scale images achieves promising results in various tasks, but prior methods for 3D hand pose pre-training have not fully utilized the potential of diverse hand images accessible from in-the-wild videos. To facilitate scalable pre-training, we first prepare an extensive pool of hand images from in-the-wild videos and design our pre-training method with contrastive learning. Specifically, we collect over 2.0M hand images from recent human-centric videos, such as 100DOH and Ego4D. To extract discriminative information from these images, we focus on the similarity of hands: pairs of non-identical samples with similar hand poses. We then propose a novel contrastive learning method that embeds similar hand pairs closer in the feature space. Our method not only learns from similar samples but also adaptively weights the contrastive learning loss based on inter-sample distance, leading to additional performance gains. Our experiments demonstrate that our method outperforms conventional contrastive learning approaches that produce positive pairs sorely from a single image with data augmentation. We achieve significant improvements over the state-of-the-art method (PeCLR) in various datasets, with gains of 15% on FreiHand, 10% on DexYCB, and 4% on AssemblyHands.

## Environment Set Up
Install required packages:
```bash
git clone https://github.com/ut-vision/simhand.git
cd simhand
conda env create -f environment.yml
conda activate simhand
## python -c "import torch; print(torch.__version__)" # Make sure it work!
```

## Run Experiments
### Prepare pre-training data
We are looking for a suitable way to publish the pre-training data set. Thank you! (Cooming soon)

### Define the environment variables
```bash
export BASE_PATH='<path_to_repo>'
export COMET_API_KEY=''
export COMET_PROJECT=''
export COMET_WORKSPACE=''
export PYTHONPATH="$BASE_PATH"
export DATA_PATH="$BASE_PATH/data/raw/"
export SAVED_MODELS_BASE_PATH="$BASE_PATH/data/models/simhand"
export SAVED_META_INFO_PATH="$BASE_PATH/data/models" 
```
### SiMHand Pre-training
For pre-training of SiMHand , please run through the code below. We did not search for enhancement strategies for SiMHand, and we inherited the description of PeCLR and SimCLR from the original PeCLR paper.
```bash
python src/experiments/main.py \
--experiment_type handclr_w \
--gpus 0,1,2,3,4,5,6,7 \ # 8 card pre-training
--color_jitter \    # Data Augmentation I
--random_crop \     # Data Augmentation II
--rotate \          # Data Augmentation III
--crop \            # Data Augmentation IV
-resnet_size 50 \   # ResNet size 50 or 152
--resize \
-sources ego4d \    # Pre-training Data Source: ego4d or 100doh
--datasets_scale 1m \   # Pre-training Data Size
-epochs 100 \
-batch_size 8192 \
-accumulate_grad_batches 1 \
-save_top_k 100 \
-save_period 1 \
-num_workers 24 \
--weight_type linear \  #  parameter-free adaptive weighting strategy
--joints_type augmented \
--diff_type mpjpe \     # distance caculation
--pos_neg pos_neg \     # add weight in pos or neg of Contrastive loss
```


## Citation
If you find our paper/code useful, please consider citing our work:

```
@inproceedings{
    lin2025simhand,
    title={{SiMHand}: Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training},
    author={Nie Lin and Takehiko Ohkawa and Yifei Huang and Mingfang Zhang and Minjie Cai and Ming Li and Ryosuke Furuta and Yoichi Sato},
    booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
    year={2025},
    url={https://openreview.net/forum?id=96jZFqM5E0}
}
```
