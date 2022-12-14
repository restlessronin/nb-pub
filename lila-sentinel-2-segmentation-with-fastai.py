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
# # UNET Trainer - FastAI
#
# I am using a FastAI v2 UNet model using the [`fastgs` library](https://github.com/restlessronin/fastgs/) for multi-spectral and geospatial image support.

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
print(fastai.__version__)
print(fastgs.__version__)

# %%
import albumentations as A
from fastai.vision.all import *

from fastgs.multispectral import *
from fastgs.test.io import *
from fastgs.vision.data import *
from fastgs.vision.learner import *


# %%
def get_input(stem: str) -> str:
    "Get full input path for stem"
    return "/kaggle/input/d/restlessronin/liladeeptrain/" + stem

def tile_img_name(chn_id: str, tile_num: int) -> str:
    "File name from channel id and tile number"
    return f"Sentinel20m-{chn_id}-20200215-{tile_num:03d}.png"

def get_channel_filenames(chn_ids, tile_idx):
    "Get list of all channel filenames for one tile idx"
    return [get_input(tile_img_name(x, tile_idx)) for x in chn_ids]


# %% [markdown]
# Create multispectral and mask data objects

# %%
sentinel2 = createSentinel2Descriptor()
all_raw_bands = MSData.from_files(
    sentinel2,
    ["B04","B03","B02","B05","B06","B07","B08","B8A","B11","B12","AOT"],
    [sentinel2.rgb_combo["natural_color"],["B07", "B06", "B05"],["B11", "B8A", "B08"],["B12"]],
    get_channel_filenames,
    read_multichan_files,
)

masks = MaskData.from_files("LC",get_channel_filenames,read_mask_file,["non-building","building"])

# %% [markdown]
# add some augmentations

# %%
aug = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=.5),
    A.VerticalFlip(p=.5)
])
augs = MSAugment(train_aug=aug,valid_aug=aug)

# %% [markdown]
# and then the pipeline object

# %%
allbands = FastGS(all_raw_bands,masks,augs)

# %% [markdown]
# create the data block

# %%
db = allbands.create_data_block()
#db.summary(source=all_tile_idxs, bs=8)

# %%
all_tile_idxs = [x for x in range(225)]

# %%
dl = db.dataloaders(source=all_tile_idxs, bs=8)

# %%
dl.show_batch(max_n=5,mskovl=False)

# %%
learner = allbands.create_unet_learner(dl,resnet18,pretrained=False,loss_func=CrossEntropyLossFlat(axis=1),metrics=Dice(axis=1))

# %%
lrs = learner.lr_find()

# %% [markdown]
# Short training loop for demo purposes

# %%
learner.fine_tune(5,lrs.valley)

# %%
learner.show_results(max_n=5,mskovl=False)

# %%
interp = SegmentationInterpretation.from_learner(learner)
interp.plot_top_losses(k=3,mskovl=False)
