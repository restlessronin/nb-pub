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

# %% [markdown]
# # Using fastai for multi-spectral images - the fastgs library
#
# I am using the [`fastgs` library](https://github.com/restlessronin/fastgs/) which provides multi-spectral / geospatial image support for fastai. The library is currently under active development, and I will attempt to update this notebook as the library evolves.

# %%
from __future__ import annotations

# %%
# # %pip install -Uqq torch torchvision fastgs
# # %pip show fastgs
# %pip install -Uqq fastgs

# %%
import torch
import fastai
import fastgs

print(torch.__version__)
print(torch.cuda.is_available())
print(fastai.__version__)
print(fastgs.__version__)

# %%
from fastai.vision.all import *

from fastgs.geospatial.sentinel import *
from fastgs.vision.testio import *
from fastgs.vision.data import *
from fastgs.vision.learner import *

# %%
import pandas as pd


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
    return TensorMask(img_arr)


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

def get_input_38(stem: str) -> str:
    "Get full input path for stem"
    return "../input/38cloud-cloud-segmentation-in-satellite-images/38-Cloud_training/" + stem

def leaf_img_path_38(chn_id: str, tile_id: int) -> str:
    color_name = id_to_color_name(chn_id)
    return f"train_{color_name}/{color_name}_patch_{tile_id}.TIF"

def get_input_95(stem: str) -> str:
    "Get full input path for stem"
    return "../input/95cloud-cloud-segmentation-on-satellite-images/95-cloud_training_only_additional_to38-cloud/" + stem

def leaf_img_path_95(chn_id: str, tile_id: int) -> str:
    color_name = id_to_color_name(chn_id)
    return f"train_{color_name}_additional_to38cloud/{color_name}_{tile_id}.TIF"

def get_channel_filenames(chn_ids, tile_id):
    "Get list of all channel filenames for one tile idx"
    return [get_input_95(leaf_img_path_95(x, tile_id)) for x in chn_ids]


# %%
all_raw_bands = MultiSpectral(
    MSDescriptor(["R","G","B","NIR"],[10,10,10,10],[1.0,1.0,1.0,1.0],{}),
    ["R","G","B","NIR"],
    "GT",
    [["R","G","B"],["NIR"]],
    get_channel_filenames,
    read_multichan_files,
    read_mask_file
)

# %%
mask_xform_block = TransformBlock(
    type_tfms=[
        partial(MultiSpectral.load_mask, all_raw_bands),
        AddMaskCodes(codes=["not-buildings", "buildings"]),
    ]
)

# %%
mc_xform_block = TransformBlock(
    type_tfms=[
        partial(MultiSpectral.load_image, all_raw_bands),
    ]
)

# %%
db = DataBlock(
    blocks=(mc_xform_block, mask_xform_block),
    splitter=RandomSplitter(valid_pct=0.2, seed=107),
)

# %%
patches = pd.read_csv("../input/95cloud-cloud-segmentation-on-satellite-images/95-cloud_training_only_additional_to38-cloud/training_patches_95-cloud_additional_to_38-cloud.csv")
db.summary(source=patches.name, bs=8)

# %%
dl = db.dataloaders(source=patches.name, bs=8)

# %%
dl.show_batch(max_n=5,mskovl=False)

# %%
learner = unet_learner(
    dl,resnet18,normalize=False,n_in=11,n_out=2,pretrained=True,
    loss_func=CrossEntropyLossFlat(axis=1),metrics=Dice(axis=1)
)

# %%
lrs = learner.lr_find()

# %%
learner.fine_tune(20,lrs.valley)

# %%
learner.show_results(max_n=5,mskovl=False)

# %%
interp = SegmentationInterpretation.from_learner(learner)
interp.plot_top_losses(k=3,mskovl=False)
