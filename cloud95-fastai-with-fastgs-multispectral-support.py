# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %pip install -Uqq fastgs

# %%
from __future__ import annotations

# %% [markdown]
# # Using fastai for multi-spectral images - the fastgs library
#
# I am using the [`fastgs` library](https://github.com/restlessronin/fastgs/) which provides multi-spectral / geospatial image support for fastai. The library is currently under active development, and I will attempt to update this notebook as the library evolves.
#
# The intention of this notebook is simply to demonstrate the use of fastgs to setup a multi-spectral fastai training pipeline. Other notebooks have already done a good job of explaining the data set and training methods. I assume the reader is generally familiar with fastai.
#
# First install fastgs. It should automatically ensure the correct version of fastai.

# %% [markdown]
# Check versions.

# %%
import torch
import fastai
import fastgs

print(torch.__version__)
print(fastai.__version__)
print(fastgs.__version__)

# %% [markdown]
# Import required libraries

# %%
from fastai.vision.all import *

from fastgs.multispectral import *
from fastgs.test.io import *
from fastgs.vision.data import *
from fastgs.vision.learner import *

import pandas as pd


# %% [markdown]
# ## Read Tensors
#
# Set up basic file io to load tensors from files and sets of files. Note that the mask values have to be changed to the format required by fastai training.

# %%
def read_chn_file(path: str) -> Tensor:
    "Read single channel file into tensor"
    img_arr = np.array(Image.open(path))
    return Tensor(img_arr / 10000)

def read_multichan_files(files: list(str)) -> Tensor:
    "Read individual channel tensor files into a tensor of channels"
    return torch.cat([read_chn_file(path)[None] for path in files])

def read_mask_file(path: str) -> TensorMask:
    """Read ground truth segmentation label files with values from 0 to n."""
    img_arr = np.array(Image.open(path))
    msk_arr = np.where(img_arr == 255, 1, 0)
    return TensorMask(msk_arr)


# %% [markdown]
# ## Generate file paths from Patch names
#
# First a function to go from a channel id to a channel name.

# %%
def id_to_color_name(chn_id: str) -> str:
    if chn_id == "R":
        return "red"
    elif chn_id == "G":
        return "green"
    elif chn_id == "B":
        return "blue"
    elif chn_id == "NIR":
        return "nir"
    elif chn_id == "GT":
        return "gt"
    else:
        assert false
        return None


# %% [markdown]
# Next we have methods to construct file paths for the two data sets (38 cloud and 95 cloud)
#
# UPDATE: There seems to be a problem with some of the files in this dataset. I get IO errors when I try to run the learner on all files. It seems to be better if I only use one of the two datasets (I picked cloud 38, which is smaller and also seems to result in quicker training)

# %%
def get_input_38(stem: str) -> str:
    "Get full input path for stem"
    return "../input/38cloud-cloud-segmentation-in-satellite-images/38-Cloud_training/" + stem

def leaf_img_path_38(chn_id: str, tile_id: int) -> str:
    color_name = id_to_color_name(chn_id)
    return f"train_{color_name}/{color_name}_{tile_id}.TIF"

def get_input_95(stem: str) -> str:
    "Get full input path for stem"
    return "../input/95cloud-cloud-segmentation-on-satellite-images/95-cloud_training_only_additional_to38-cloud/" + stem

def leaf_img_path_95(chn_id: str, tile_id: int) -> str:
    color_name = id_to_color_name(chn_id)
    return f"train_{color_name}_additional_to38cloud/{color_name}_{tile_id}.TIF"


# %% [markdown]
# We need to be able to check which set the patch belongs to

# %%
#only95names = pd.read_csv("../input/95cloud-cloud-segmentation-on-satellite-images/95-cloud_training_only_additional_to38-cloud/training_patches_95-cloud_additional_to_38-cloud.csv").name
#only95 = frozenset(only95names)

# %%
def get_channel_filenames(chn_ids, tile_id):
    "Get list of all channel filenames for one tile idx"
    return [get_input_38(leaf_img_path_38(x, tile_id)) for x in chn_ids]


# %% [markdown]
# Next we create a channel descriptor for the images. This lists all the possible channels ids for our multi-spectral source.

# %%
landsatDesc = MSDescriptor.from_bands(["R","G","B","NIR"])

# %% [markdown]
# Next create the fastgs helper class. The first parameter is a list of the channel ids in our example in the order in which they will be loaded. Next is the id of the mask.
#
# The third parameter is what allows multi-spectral visualization. It lists sets of 3 channels (or 1 channel) which can be used to produce "false-colour" RGB (or monochrome) images from the channels in our input.
#
# In this example, each ms tensor will produce two images, one "false-colour" with NIR, G, B (instead of R,G,B), and another using (R,G,B). We could also choose to display individual bands as monochrome images.
#
# Set up data wrapper classes for images and masks

# %%
landsat_imgs = MSData.from_files(
    landsatDesc,
    ["R","G","B","NIR"],
    [["NIR","G","B"],["R","G","B"]],
    get_channel_filenames,
    read_multichan_files
)

landsat_msks = MaskData.from_files("GT",get_channel_filenames,read_mask_file,["non-cloud","cloud"])

# %% [markdown]
# add some augmentations

# %%
import albumentations as A

# %%
aug = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=.5),
    A.VerticalFlip(p=.5)
])
augs = MSAugment.from_augs(train_aug=aug,valid_aug=aug)

# %% [markdown]
# Now create the master wrapper class.

# %%
landsat = FastGS.for_training(landsat_imgs,landsat_msks,augs)

# %% [markdown]
# the wrapper class knows how to create a fastai DataBlock.

# %%
db = landsat.create_data_block()

# %% [markdown]
# We will only use 1000 images to create the DataLoader, since the purpose of this notebook is to demonstrate the fastgs pipeline.

# %%
#all_ex_95 = pd.read_csv("/kaggle/input/95cloud-cloud-segmentation-on-satellite-images/95-cloud_training_only_additional_to38-cloud/training_patches_95-cloud_additional_to_38-cloud.csv").name
#nonempty_95 = pd.read_csv("/kaggle/input/95cloud-cloud-segmentation-on-satellite-images/95-cloud_training_only_additional_to38-cloud/training_patches_95-cloud_nonempty.csv").name
#ne_ex_95 = list(set(all_ex_95) & set(nonempty_95))
ne_ex_95 = pd.read_csv("/kaggle/input/38cloud-cloud-segmentation-in-satellite-images/training_patches_38-cloud_nonempty.csv").name
print(len(ne_ex_95))
#nonempty = pd.read_csv("../input/95cloud-cloud-segmentation-on-satellite-images/95-cloud_training_only_additional_to38-cloud/training_patches_95-cloud_nonempty.csv").name
small=ne_ex_95[0:1000]

#db.summary(source=nonempty400, bs=8)

dl = db.dataloaders(source=small, bs=8)

dl.show_batch(max_n=5,mskovl=False)

# %% [markdown]
# the wrapper class also creates the unet learner

# %%
learner = landsat.create_learner(dl)

# %%
lrs = learner.lr_find()

# %% [markdown]
# Keep the training loop short to avoid IO errors.

# %%
learner.fine_tune(5,lrs.valley)

# %%
learner.show_results(max_n=5,mskovl=False)

# %%
interp = SegmentationInterpretation.from_learner(learner)
interp.plot_top_losses(k=3,mskovl=False)
