# SiMHand
This is the official implementation of our ICLR 2025 paper "[SiMHand: Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training](https://openreview.net/forum?id=96jZFqM5E0)". Hop to share our work in Singapore 🇸🇬 ～!!

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