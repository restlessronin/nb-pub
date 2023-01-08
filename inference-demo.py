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
#     display_name: Python [conda env:lila-deep] *
#     language: python
#     name: conda-env-lila-deep-py
# ---

# %%
# #%pip install -Uqq fastgs

# %%
from __future__ import annotations

# %%
import fastgs

print(fastgs.__version__)

# %%
from fastai.vision.all import *
from fastgs.multispectral import *
from fastgs.vision.data import *
from fastgs.vision.core import *
from fastgs.vision.learner import *

# %%
# import packages used for deaping with spatial raster data
import rasterio as rio
from tqdm.auto import tqdm
import os
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt

# %%
# set the path to the test sentinel 2 scene
root = '/Users/Shared/kaggle/'
l1c_safe_folder = Path(root + 'input/sentinel-2-l1c-cloud-example/S2B_MSIL1C_20211127T003659_N0301_R059_T55KBU_20211127T015218/S2B_MSIL1C_20211127T003659_N0301_R059_T55KBU_20211127T015218.SAFE')

# %%
# set the location to store our inference data
working = root + 'working/'
output_dir = Path(working + 's2_wokring_v2')
output_dir.mkdir(exist_ok=True)

# %%
# set up some vars for storing data
scene_name = l1c_safe_folder.name
output_path = output_dir/scene_name
tile_folder = output_path/'tiles'


# %%
# search for all tif files in img_dir, these are our tiles
def get_inf_files_paths(img_dir):
    return list(img_dir.rglob('*.tif'))


# %%
# get a count off all the tiles
tiles = get_inf_files_paths(tile_folder)
len(tiles)


# %%
# open image with rasterio and convert it into float range
def get_ms_np_rio(bands,img_path):
    return rio.open(img_path).read()/2**16

def get_ms_tensor_rio(bands,img_path):
    return TensorImageMS(get_ms_np_rio(bands, img_path))


# %%
required_bands = ["B04", "B03", "B02", "B08", "B10"]
mask_codes=['UNDEFINED','CLEAR','CLOUD SHADOW','SEMI TRANSPARENT CLOUD','CLOUD','MISSING']


# %%
# create a fastgs data descriptor for our tiles
def createSentinel2L1CDescriptor_inf() -> MSDescriptor:
    return MSDescriptor.from_all(
        required_bands,
        [20,20,24,15,20],
        [10,10,10,10,60],
        {# https://gisgeography.com/sentinel-2-bands-combinations/
            "natural_color": ["B04","B03","B02"]
        }
    )

S2L1C_desc = createSentinel2L1CDescriptor_inf()

# %%
# create a fastgs MSData object 
S2L1C_bands_55 = MSData.from_loader(
    S2L1C_desc,
    required_bands,
    [S2L1C_desc.rgb_combo["natural_color"],["B08"],["B10"]],
    get_ms_tensor_rio
)

fgs = FastGS.for_inference(S2L1C_bands_55,mask_codes)

# %%
# open the first image and check the stats
open_img = S2L1C_bands_55.load_image(tiles[0])
print(f'Shape {open_img.shape}\nMin val {open_img.min()}\nMax val {open_img.max()}')

# %%
open_img.show()

# %%
# build a fastai DataBlock
db = DataBlock(S2L1C_bands_55.create_xform_block())

# %%
#db.summary(source=tiles)

# %%
# build a fastai dataloader
dl = db.dataloaders(source=tiles, bs=1)

# %%
#dl.show_batch(max_n=5)

# %%
learner = fgs.load_learner(working + "models/S2_cloud_model",dl)

# %%
dl = learner.dls.test_dl(test_items=tiles, bs=1)

# %%
# show a batch, this does not work, not sure why?
dl.show_batch(max_n=5,mskovl=False)
