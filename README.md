<h1 align="center">
  Learning Discriminative Geometry for Drifting Models
</h1>

<p align="center">
  <a href="https://zhangdoudou.github.io/">Doudou Zhang</a><sup>1,2</sup>
  &nbsp;&nbsp;&nbsp;
  <a href="https://sites.google.com/view/wenwen-hou">Wenwen Hou</a><sup>1,2</sup>
  &nbsp;&nbsp;&nbsp;
  <a href="https://yilinchen1205.github.io/">Yilin Chen</a><sup>1,2</sup>
  &nbsp;&nbsp;&nbsp;
  <a href="https://livreq.github.io/">Qi Chen</a><sup>1,2,*</sup>
</p>

<p align="center">
  <sup>1</sup>ELLIS Institute Finland
  &nbsp;&nbsp;&nbsp;
  <sup>2</sup>Aalto University
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.04703">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv Paper">
  </a>
</p>

<p align="center">
  <img src="assets/figure1.png" width="100%">
</p>

## News

- **[2026-10]** Code released.

## Abstract

Recently proposed **Drifting Models** shift iterative distribution refinement from inference to training, enabling effective one-step generation. However, their performance on complex image datasets depends strongly on the representation used to construct the drifting field: pixel-space drifting performs poorly, whereas pretrained feature spaces substantially improve sample quality for reasons that remain unclear.

We trace this gap to the **discriminative geometry of the representation**, which determines sample weighting in kernel density estimation (KDE) and, consequently, drift. We introduce **persistent representation learning**, which continuously learns a more discriminative representation geometry as the generator evolves across batches.

We further establish a **current-step gradient equivalence between the KDE ratio loss and drift regression loss** under matched conditions, connecting density-ratio-based generator optimization to empirical drifting and motivating direct control of the drifting velocity.

Across multiple datasets, our method learns effective discriminative representations directly from pixels and reduces FID by approximately **82–95%** over the original pixel-space Drifting Models, without pretrained encoders. Adapting pretrained representations and applying velocity clipping provide further gains.

## Files

| File | Content |
|---|---|
| `unet.py` | U-Net generator mapping Gaussian noise to images in a single forward pass |
| `encoder.py` | Residual encoder $E_\varphi(x)=x+R_\varphi(x)$ with zero-initialized output projection, and the feature groups extracted from its intermediate maps |
| `kde_loss.py` | KDE classification objective used to update the encoder |
| `drift.py` | Drifting field in the learned representation, velocity clipping, and drift regression loss |
| `distributed.py` | Collective operations for gathering KDE anchors across GPUs |
| `train.py` | Alternating encoder and generator updates |
| `sample.py` | Sample generation from a checkpoint |

## Environment Setup

```bash
conda create -n pdrift python=3.11 -y
conda activate pdrift
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Training

Each training step performs one encoder update on a fresh real and generated batch, followed by one generator update on another fresh batch. The KDE anchors are gathered across all GPUs.

All reported experiments use 4 NVIDIA GH200 GPUs with 96 GB of GPU memory each. The training set is downloaded automatically by torchvision to `--data-root`.

```bash
torchrun --nproc_per_node 4 train.py --dataset cifar10 --data-root ./data --output-dir ./runs/cifar10
torchrun --nproc_per_node 4 train.py --dataset cifar100 --data-root ./data --output-dir ./runs/cifar100
torchrun --nproc_per_node 4 train.py --dataset svhn --data-root ./data --output-dir ./runs/svhn
```

## Sampling

```bash
python sample.py --checkpoint ./runs/cifar10/checkpoints/step0050000.pt --output-dir ./samples/cifar10 --num-samples 50000
```

FID can be computed between the generated samples and the training set with a standard implementation such as `pytorch-fid`.

## Citation

If you find our work useful for your research, please consider citing:

```bibtex
@article{zhang2026learning,
  title   = {Learning Discriminative Geometry for Drifting Models},
  author  = {Zhang, Doudou and Hou, Wenwen and Chen, Yilin and Chen, Qi},
  journal = {arXiv preprint arXiv:2610.04703},
  year    = {2026}
}
```

## Acknowledgements

This codebase builds on [DriftXpress](https://github.com/Mortrest/DriftXpress). We thank the authors for releasing their code.

## License

This project is released under the [MIT License](LICENSE).
