# Complete AI-Enabled Echocardiography Interpretation with Multitask Deep Learning

*[Gregory Holste](https://gholste.me), Evangelos K. Oikonomou, Márton Tokodi, Attila Kovács, Zhangyang Wang, [Rohan Khera](https://www.cards-lab.org/rohan-khera)*

**JAMA** (June 23, 2025)

[[`Paper`](https://jamanetwork.com/journals/jama/fullarticle/2835630?guestAccessKey=27a79d8c-caca-4f4a-ad10-70c202cdfe4e&utm_source=twitter&utm_medium=social_jama&utm_term=17313965533&utm_campaign=article_alert&linkId=833098185)] [[`Preprint`](https://www.medrxiv.org/content/10.1101/2024.11.16.24317431v2)] [[`Project Page`](https://www.cards-lab.org/panecho)]

-----

<p align=center>
    <img src=content/panecho.png height=500>
</p>

PanEcho is a view-agnostic, multi-task model that performs 39 reporting tasks from echocardiogram videos. The model consists of an image encoder, temporal frame Transformer, and task-specific output heads. Since PanEcho is view-agnostic, multi-view echocardiography can be leveraged to integrate information across views. At test time, predictions are aggregated across echocardiograms acquired during the same study to generate study-level predictions for each task.

## Model Usage

The pretrained model is available with a simple one-line import using `PyTorch`.
```
import torch

# Import PanEcho
model = torch.hub.load('CarDS-Yale/PanEcho', 'PanEcho', force_reload=True)

# Demo inference on random video input
x = torch.rand(1, 3, 16, 224, 224)
print(model(x))
```

`Input`: PyTorch Tensor of shape `(batch_size, 3, 16, 224, 224)` containing a 16-frame echocardiogram video clip at 224 x 224 resolution that has been ImageNet-normalized. See our [data loader](https://github.com/CarDS-Yale/PanEcho/blob/34f611db51fd78f6bb1f7a6ef7ab59736afda1d2/src/dataset.py#L389) for implementation details, though you are free to preprocess however you like.

`Output`: Dictionary whose keys are the tasks (**see [here](https://github.com/CarDS-Yale/PanEcho/blob/main/content/tasks.md) for breakdown of all tasks**) and values are the predictions. The model outputs class-wise probabilities for classification tasks and direct estimates for regression tasks.

### Setting Video Length

If you want to run inference on video clips of more or less than 16 frames, you can control this through the `clip_len` argument:
```
import torch

# Import PanEcho
model = torch.hub.load('CarDS-Yale/PanEcho', 'PanEcho', force_reload=True, clip_len=32)

# Demo inference on random 32-frame video input
x = torch.rand(1, 3, 32, 224, 224)
print(model(x))
```

We recommend using 16-frame video clips when using PanEcho directly for inference in accordance with how the model was trained and evaluated. However, feel free to experiment with video length, particularly if you are fine-tuning PanEcho on your own data (see "Transfer Learning" below).

### Subsetting Task Heads

If you want to use PanEcho but are only interested in a subset of tasks, you can easily select your tasks of interest as follows. *Note*: Be sure to consult [this table](https://github.com/CarDS-Yale/PanEcho/blob/main/content/tasks.md) to select the desired task names.
```
# Import PanEcho, but remove unwanted task heads
tasks = ['EF', 'LVSystolicFunction', 'AVStenosis']
model = torch.hub.load('CarDS-Yale/PanEcho', 'PanEcho', force_reload=True, tasks=tasks)

# Demo inference on random video input
x = torch.rand(1, 3, 16, 224, 224)
print(model(x))  # predictions only for EF, LVSystolicFunction, & AVStenosis
```

### Transfer Learning

To fine-tune PanEcho on your own data, we provide two options:

#### 1) For video-based (3D) modeling, use the entire pretrained PanEcho backbone as follows:
```
# Import PanEcho backbone w/o task-specific output heads
model = torch.hub.load('CarDS-Yale/PanEcho', 'PanEcho', force_reload=True, backbone_only=True)

# Demo inference on random video input
x = torch.rand(1, 3, 16, 224, 224)
print(model(x))  # (1, 768) video embedding
```

`Input`: PyTorch Tensor of shape `(batch_size, 3, 16, 224, 224)` containing a 16-frame echocardiogram video clip, as described earlier.

`Output`: PyTorch Tensor of shape `(batch_size, 768)` containing an embedding (learned representation) of the video.

#### 2) For image-based (2D) modeling, use the pretrained image encoder of PanEcho as follows:
```
# Import PanEcho image encoder w/o temporal Transformer or task-specific output heads
model = torch.hub.load('CarDS-Yale/PanEcho', 'PanEcho', force_reload=True, image_encoder_only=True)

# Demo inference on random image input
x = torch.rand(1, 3, 224, 224)
print(model(x))  # (1, 768) image embedding
```

`Input`: PyTorch Tensor of shape `(batch_size, 3, 224, 224)` containing an echocardiogram image (one video frame).

`Output`: PyTorch Tensor of shape `(batch_size, 768)` containing an embedding (learned representation) of the image.

## Requirements & Setup for Reproducibility

*Note*: This repository is intended for use on a Linux machine with access to at least one NVIDIA GPU. While PanEcho was trained using NVIDIA A100 GPUs, the codebase is expected to function with virtually any recent NVIDIA GPUs. Specifically, PanEcho was developed on a Linux machine running Ubuntu 22.04.4 LTS with 8 NVIDIA A100 GPUs. All code and the following demo was also tested on an older Linux machine running Ubuntu 20.04.6 LTS with 4 NVIDIA 3090 GPUs with identical results.

### Install Packages
On a Linux machine with [Anaconda](https://docs.anaconda.com/) or [Miniconda](https://docs.anaconda.com/miniconda/) installed, run the following at the command line. Installation of required packages may take a few minutes.
```
# Navigate to the repository
cd PanEcho/  # if you are one directory above

# Install required packages
conda env create -f content/panecho.yml  # this can take up to 5-10 minutes

# Activate environment
conda activate panecho
```

## Experiments

While the YNHHS cohort used to develop PanEcho is not currently available for sharing due to IRB restrictions, the following commands were used to produce the results in the paper:

```
# Navigate to proper directory
cd src/

# Train PanEcho with distributed training across 8 GPUs
torchrun --nproc_per_node=8 train.py --output_dir panecho --model_name frame_transformer --arch convnext_tiny --fc_dropout 0.25 --normalization imagenet --augment --reduce_lr 3 --amp

### EchoNet-Dynamic fine-tuning experiments ###
# Fine-tune PanEcho on EchoNet-Dynamic
torchrun --nproc_per_node=8 train_echonetdynamic.py --model_dir_path panecho --output_dir echonetdynamic_ft --model_name frame_transformer --arch convnext_tiny --normalization imagenet --amp

# Fine-tune a randomly initialized PanEcho architecture
torchrun --nproc_per_node=8 train_echonetdynamic.py --output_dir echonetdynamic_ft --rand_init --model_name frame_transformer --arch convnext_tiny --normalization '' --amp

# Fine-tune an ImageNet-pretrained PanEcho architecture
torchrun --nproc_per_node=8 train_echonetdynamic.py --output_dir echonetdynamic_ft --model_name frame_transformer --arch convnext_tiny --normalization imagenet --amp

# Fine-tune an EchoCLIP-pretrained PanEcho architecture
torchrun --nproc_per_node=8 train_echonetdynamic.py --output_dir echonetdynamic_ft --model_name frame_transformer --arch echo-clip --normalization echo-clip --amp

# Fine-tune a Kinetics-400-pretrained 3DResNet18
torchrun --nproc_per_node=8 train_echonetdynamic.py --output_dir echonetdynamic_ft --model_name 3dresnet18 --normalization kinetics --amp

### EchoNet-Pediatric fine-tuning experiments ###
# Simply replace "echonetdynamic" with "echonetpediatric" in the previous five commands
```

## Citation

MLA:
```
Holste, Gregory, et al. "Complete AI-Enabled Echocardiography Interpretation With Multitask Deep Learning." JAMA, 2025, p. 2025.8731.
```

BibTeX:
```
@article{holste2025complete,
    author = {Holste, Gregory and Oikonomou, Evangelos K. and Tokodi, Márton and Kovács, Attila and Wang, Zhangyang and Khera, Rohan},
    title = {Complete AI-Enabled Echocardiography Interpretation With Multitask Deep Learning},
    journal = {JAMA},
    year = {2025},
    month = {06},
    issn = {0098-7484},
    doi = {10.1001/jama.2025.8731},
    url = {https://doi.org/10.1001/jama.2025.8731},
}
```

## Contact

Reach out to Greg Holste (gholste@utexas.edu) and Rohan Khera (rohan.khera@yale.edu) with any questions.
