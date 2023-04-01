# Learning Domain-invariant Representations from Text for Domain Generalization

## Abstract
Domain generalization (DG) aims to transfer the knowledge learned in the source domain to the unseen target domain. Most DG methods focus on studying how to learn domain-invariant representations that remain invariant across different domains. For humans, we tend to use the same word or text to describe images from different domains but of the same category. Therefore, text can be considered a natural domain-invariant representation. Inspired by this, we study how to introduce text representations into domain generalization tasks. Specifically, we use the text representations from the CLIP text encoder to guide the image representation learning of the visual model. Meanwhile, we also find that the CLIP representations have weak discriminability and overfit some domains. To alleviate these problems, we combine the text representation regularization loss with standard image-level supervised loss. Our proposed method is simple yet efficient, and achieves competitive performance compared with the existing state-of-the-art methods on five standard DG datasets.
<p align="center">
    <img src="./assets/method.png" width="90%" />
</p>

---

Note that this project is built upon [DomainBed@3fe9d7](https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414) and [SWAD](https://github.com/khanrc/swad).


## Preparation

### Dependencies

```sh
pip install -r requirements.txt
```

### Datasets

```sh
python -m domainbed.scripts.download --data_dir=/data/
```

### Environments

Environment details used for our study.

```
Python: 3.7.13
PyTorch: 1.12.0+cu113
Torchvision: 0.13.0+cu113
CUDA: 11.3
CUDNN: 8.2
NumPy: 1.21.5
PIL: 9.0.1
```

## How to Run

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain.

```sh
python train_all.py exp_name --dataset PACS --data_dir /data/
```

- PACS

```
CUDA_VISIBLE_DEVICES=0 python train_all.py PACS --dataset PACS --deterministic \
--trial_seed 0 --checkpoint_freq 100 --steps 5000 --data_dir $data_dir --algorithm Contrast \
--contrast_w 0.1 --lr 5e-5
```

- VLCS

```
CUDA_VISIBLE_DEVICES=0 python train_all.py VLCS --dataset VLCS --deterministic \
--trial_seed 0 --checkpoint_freq 200 --steps 5000 --data_dir $data_dir --algorithm Contrast \
--contrast_w 0.01 --lr 1e-6
```

- OfficeHome

```
CUDA_VISIBLE_DEVICES=0 python train_all.py OH --dataset OfficeHome --deterministic \
--trial_seed 0 --checkpoint_freq 200 --steps 5000 --data_dir $data_dir --algorithm Contrast \
--contrast_w 1 --lr 5e-5
```

- TerraIncognita

```
CUDA_VISIBLE_DEVICES=0 python train_all.py TR0 --dataset TerraIncognita --deterministic \
--trial_seed 0 --checkpoint_freq 200 --steps 5000 --data_dir $data_dir --algorithm Contrast \
--contrast_w 1 --lr 5e-5
```

- DomainNet

```
CUDA_VISIBLE_DEVICES=0 python train_all.py DomainNet --dataset DomainNet --deterministic \
--trial_seed 0 --checkpoint_freq 1000 --steps 15001 --data_dir $data_dir --algorithm Contrast \
--contrast_w 0.1 --lr 5e-5
```