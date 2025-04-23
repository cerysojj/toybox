# Infant-Inspired Deep Learning for Robust Visual Recognition

This repository contains code for experiments on training CNNs with infant-inspired curricula. This project examines how a training CNNs with a developmentally-inspired blurry-to-clear curriculum shapes robustness to blur across domains (faces vs. objects) and classification levels (instance vs. category).

---

## Datasets

### Toybox

The [Toybox dataset](https://aivaslab.github.io/toybox/) (Wang et al., 2018) contains egocentric videos of 360 objects across 12 everyday categories.

In my preliminary experiments, I do not use bounding boxes. But in my final experiments, I use the version with bounding box annotations to control object scale. Two configurations were used:

- **Instance-level**: Each object is a unique class (360 total).
  - **Train**: 6 rotation videos per object (~371 images/class).
  - **Val/Test**: Sampled from hodgepodge videos (frames trimmed to remove canonical views).

- **Category-level**: 12 object categories, each with 30 instances.
  - **Train**: 24 objects/category (rotation videos only).
  - **Val**: Objects #28–30.
  - **Test**: 3 visually distinct objects/category.
  - Max 3,000 training images/category.

### ImageNet

- **ImageNet-12**: Curated from [ImageNet](https://www.image-net.org/) and [MS-COCO](https://cocodataset.org/) to match Toybox categories.
  - **Train**: 1,500 images/class
  - **Val/Test**: 100 images/class

- **[ImageNet-100](https://www.kaggle.com/datasets/ambityga/imagenet100)**: A 100-class subset of ImageNet.
  - **Train**: 1,300 images/class
  - **Val**: 50 images/class

### Faces

Face images were collected from the [FaceScrub dataset](https://vintage.winklerbros.net/facescrub.html) (Ng & Winkler, 2014).

- Classes with fewer than 80 images were excluded.
- Final dataset: 297 face identities.
  - **Train**: ~80 images/class
  - **Val/Test**: ~10 images/class

---

## Environment Setup

For users running the scripts on SLURM for the preliminary experiments, please set up your environment using the provided `requirements.txt` file. This file is tailored for Linux-64 systems and was generated using conda 24.9.2.

To create a Conda environment from this file, run:

```bash
conda create --name <env_name> --file requirements.txt
```

---

## Preliminary Experiments

Preliminary experiments (e.g., blur and colour curricula) were run on the University of Edinburgh School of Informatics MLP research cluster using SLURM.

To submit a job from `prelim/src`, use:

```bash
sbatch --time=08:00:00 --gres=gpu:1 scripts/run_job.sbatch \
    --model AlexNet \
    --dataset toybox \
    --epochs 10 \
    --learning_rate 0.01 \
    --source_data_dir /path/to/data
```

### Argument Options
You can customize the model, dataset, and training mode by adjusting the arguments in your SLURM command:

| Use Case	| --model | --dataset	| Additional Arguments | 
| ----------| -----------------| --------------------- | ------------------- |
| Standard Toybox training using AlexNet	| AlexNet	| toybox	| 
| Blurry Toybox training	| AlexNet	| toybox_blur	| 
| Grayscale Toybox training	| AlexNet	| toybox_grayscale	| 
| Training on MNIST	| _any_	| MNIST	| 
| Use ResNet-18 architecture	| ResNet18	| _any_	|
| Resume training from checkpoint  | _any_  | _any_  | `--continue_from_epoch <EPOCH>`<br>`--source_output_dir <PATH_TO_PREVIOUS_OUTPUT>` |

---

## Main Experiments
All main experiments were implemented in Jupyter notebooks and executed on Google Colab, using an A100 GPU.

Notebooks for running AlexNet training are located in the `main/notebooks/model_training/alexnet` directory.

The `main/src` directory contains Python scripts used by the model training notebooks, including model definitions, custom dataset classes, and utility functions.

Notebooks for evaluating trained CNNs—such as measuring robustness to blur, calculating receptive field size, and analysing spatial frequency preferences—are located in the `main/notebooks/figures` directory.

Training logs and metrics are saved in the `main/output` directory.
