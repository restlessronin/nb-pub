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

# %%
import fastgs

print(fastgs.__version__)

# %%
from fastai.vision.all import *
from fastgs.multispectral import *
from fastgs.vision.data import *
from fastgs.vision.core import *
from fastgs.vision.learner import *

# %% [markdown]
# ## Reading NetCDF files
#
# This dataset is a little different from others on which I have run fastai. It contains data for all the channels in a NetCDF format. I am not very familiar with the format, so this is an exploration.

# %%
import netCDF4 as nc

# %%
data_path = Path('/kaggle/input/kappaset-sentinel2-kappazeta-cloudshadow-subset/KappaSet_subset/')

# %% [markdown]
# Search for all netcdf files within path, and print the file count.

# %%
all_data = list(data_path.rglob('*.nc'))
all_training_data = []
for i in all_data:
    if 'test' not in str(i):
        all_training_data.append(i)
len(all_training_data)

# %% [markdown]
# Open up one of the files and see the labels inside.

# %%
labels = list(nc.Dataset(all_training_data[0]).variables.keys())
labels

# %% [markdown]
# Make a list with the bands we want to use.

# %%
raw_bands_labels = []
for i in range(1,13):
    raw_bands_labels.append(f'B{str(i).zfill(2)}')
raw_bands_labels

# %% [markdown]
# Make a dict of dicts to store the highest and lowest values from each band

# %%
band_min_max_tracker = {}
for i in raw_bands_labels:
    band_min_max_tracker[i] = {'min':float('inf'),
                               'max':0}
band_min_max_tracker

# %%
band_min_max_tracker.keys()

# %%
from tqdm.auto import tqdm

# %% [markdown]
# Loop over each training data chip, open it and for each band grab the lowest and highest non 0 value, if these values are lower and higher than the previous values store them. We will do this for tthe entire dataset, however if this dataset was large we could just grab a random sample.

# %%
for chip in tqdm(all_training_data):
    open_chip = nc.Dataset(chip)
    for raw_band in band_min_max_tracker.keys():
        try:
            chip_band = open_chip[raw_band][:]
            chip_min = np.min(chip_band[np.nonzero(chip_band)])
            chip_max = np.max(chip_band[np.nonzero(chip_band)])
            if chip_min < band_min_max_tracker[raw_band]['min']:
                band_min_max_tracker[raw_band]['min'] = chip_min
            if chip_max > band_min_max_tracker[raw_band]['max']:
                band_min_max_tracker[raw_band]['max'] = chip_max
        except Exception as ex:
            print(ex)

# %% [markdown]
# We can now strech our data using these values to pull out as much contract as possible

# %%
band_min_max_tracker

# %% [markdown]
# Open a netcdf format file as a dataset

# %%
fn = "/kaggle/input/kappaset-sentinel2-kappazeta-cloudshadow-subset/KappaSet_subset/April/T34UFA_20200401T093029_tile_3_9.nc"
ds = nc.Dataset(fn)
print(ds)

# %% [markdown]
# We can see from the metadata that it appears to contain variables B01 ... etc that correspond to sentinel bands.
#
# Let us look at the metadata for 5 bands

# %%
bands=["B04","B03","B02","B08","B10"]
v = [ds.variables[x] for x in bands]
print(ds["B04"])

# %% [markdown]
# Now let's zoom in on one band "B04"

# %%
ds["B04"][:]


# %% [markdown]
# This is a numpy array, so we create a function to load it as a tensor. 
# *Note that I don't understand exactly how the fill value works to mark missing values, so I'm ignoring this while reading the files*

# %%
def get_tensor(ds, band):
    return Tensor(ds[band][:])


# %%
get_tensor(ds, "B04")

# %% [markdown]
# Now we create a function that loads a multi-spectral tensor for the band list we created earlier

# %%
iid = "/kaggle/input/kappaset-sentinel2-kappazeta-cloudshadow-subset/KappaSet_subset/April/T34UFA_20200401T093029_tile_3_9.nc"

def get_ms_tensor(bands, img_id):
    ds = nc.Dataset(img_id)
    return torch.cat([get_tensor(ds,b)[None] for b in bands])

get_ms_tensor(bands, iid)

# %% [markdown]
# This is the basic function that we need to create our fastai higher level wrappers.

# %%
sentinel2 = createSentinel2Descriptor()
five_bands = MSData.from_loader(
    sentinel2,
    bands,
    [["B04", "B03", "B02"],["B08"],["B10"]],
    get_ms_tensor
)

# %%
five_bands_img = five_bands.load_image(iid)

# %%
five_bands_img.show()

# %% [markdown]
# It appears that the brightening factors need to be higher for the RGB image, let's try 5 monochrome images.

# %%
bands_55 = MSData.from_loader(
    sentinel2,
    bands,
    [["B04"], ["B03"], ["B02"],["B08"],["B10"]],
    get_ms_tensor
)

# %%
bands_55_img = bands_55.load_image(iid)

# %%
bands_55_img.show()


# %% [markdown]
# This is a little bit better, but we should probably calculate brightening factors after an analysis of actual data values of the RGB channels for some a number of images.
#
# We can also make a custom Descriptor with specific scaling factors for this dataset

# %%
def createSentinel2KappaSetDescriptor() -> MSDescriptor:
    return MSDescriptor.from_all(
        ["TCI_R", "TCI_G", "TCI_B", "B04", "B03", "B02", "B08", "B05", "B8A", "B06", "B11", "B12", "B09", "B10", "B01", "Label", "B07"],
        [0.0039,0.0039,0.0039,20,20,24,15,30,15,10,10,15,25,20,30,0.2,10],
        [10,10,10,10,10,10,10,20,20,20,20,20,60,60,60,10,20],
        {# https://gisgeography.com/sentinel-2-bands-combinations/
            "natural_color": ["B04","B03","B02"],
            "color_infrared": ["B08","B04","B03"],
            "short_wave_infrared": ["B12","B8A","B04"],
            "agriculture": ["B11","B08","B02"],
            "geology": ["B12","B11","B02"],
            "bathymetric": ["B04","B03","B01"]
        }
    )


# %%
sentinel2_KappaSet = createSentinel2KappaSetDescriptor()

# %%
five_bands_imgs = MSData.from_loader(
    sentinel2_KappaSet,
    bands,
    [["B04", "B03", "B02"],["B08"],["B10"]],
    get_ms_tensor
)

# %%
five_bands_custom_descriptor_img = five_bands_imgs.load_image(iid)

# %%
five_bands_custom_descriptor_img.show();


# %%
def read_mask(msk_id, img_id):
    ds = nc.Dataset(img_id)
    return TensorMask(ds[msk_id][:])


# %%
read_mask("Label", iid)

# %%
mask_codes=['UNDEFINED','CLEAR','CLOUD SHADOW','SEMI TRANSPARENT CLOUD','CLOUD','MISSING']

# %%
masks = MaskData.from_loader("Label", read_mask, mask_codes)

# %%
msk = masks.load_mask(iid)
msk.show()

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
# create fastgs wrapper

# %%
fgs = FastGS.for_training(five_bands_imgs, masks, augs)

# %% [markdown]
# the datablock

# %%
db = fgs.create_data_block()

# %%
dl = db.dataloaders(source=all_training_data, bs=8)

# %%
dl.show_batch(max_n=5,mskovl=False)

# %%
learner = fgs.create_learner(dl)

# %%
lrs = learner.lr_find()

# %%
learner.fit_one_cycle(25,lrs.valley)

# %%
# save model to disk
learner.save("S2_cloud_model")

# %%
learner.show_results(max_n=5,mskovl=False)

# %%
interp = SegmentationInterpretation.from_learner(learner)
interp.plot_top_losses(k=3,mskovl=False)

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
l1c_safe_folder = Path('/kaggle/input/sentinel-2-l1c-cloud-example/S2B_MSIL1C_20211127T003659_N0301_R059_T55KBU_20211127T015218/S2B_MSIL1C_20211127T003659_N0301_R059_T55KBU_20211127T015218.SAFE')

# %%
# set the path to the vrt file, normaly we would make this with gdal, but installing gdal on Kaggle is difficult
vrt_path = Path('/kaggle/input/sentinel-2-l1c-cloud-example/Stack.vrt')

# %%
# set the location to store our inference data
output_dir = Path('/kaggle/working/s2_wokring_v2')
output_dir.mkdir(exist_ok=True)

# %%
# set up some vars for storing data
scene_name = l1c_safe_folder.name
output_path = output_dir/scene_name
tile_folder = output_path/'tiles'
tile_folder.mkdir(exist_ok=True,parents=True)

# %%
# set the tile size in pixels
tile_size_px = [512,512]

# %%
# these are the bands we need for the model
required_bands = ["B04", "B03", "B02", "B08", "B10"]

# %%
# search fo the bands that we need
bands_to_tile_paths = []
for band in required_bands:
     bands_to_tile_paths.append(str(list(l1c_safe_folder.rglob(f'*IMG_DATA/*{band}.jp2'))[0]))
bands_to_tile_paths


# %%
# func to pad bottom and right most tiles to the full tile size
def pad_array(array):
    bands, y_shape, x_shape  = array.shape
    x_missing =  tile_size_px[0] - x_shape
    y_missing =  tile_size_px[1] - y_shape
    if x_missing or y_missing:
        array = np.pad(array,[(0, 0),(0, y_missing), (0, x_missing)])
    return array


# %%
S2_scene = rio.open(vrt_path)
S2_scene_meta = S2_scene.profile
S2_scene_meta

# %%
# make a list of tiles by looping over raster height and width of the input scene
tiles = []
left = 0
tile_count = 0
while left < S2_scene_meta['width']:
    top=0
    while top < S2_scene_meta['height']:
        tile_count += 1
        name = f'part_{tile_count}.tif'
        export_path = tile_folder/name
        
        tiles.append([left,top,export_path])
        top += tile_size_px[1]
    left += tile_size_px[0]
print(len(tiles))
tiles[1]

# %%
# cut out each tile from list
for tile in tqdm(tiles):
    win = rio.windows.Window(tile[0], tile[1], tile_size_px[0], tile_size_px[1])
    win_data = S2_scene.read(window=win)
    data_shape = win_data.shape
    if data_shape[1] != tile_size_px[0] or data_shape[2] != tile_size_px[1]:
        win_data = pad_array(win_data)
    win_transform = S2_scene.window_transform(win)
    meta = S2_scene.meta
    meta['driver'] = 'GTiff'
    meta['transform'] = win_transform
    meta['width'] = win_data.shape[2]
    meta['height'] = win_data.shape[1]
    with rio.open(tile[2], 'w', **meta) as dst:
        dst.write(win_data)

# %%
meta


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
    return Tensor(get_ms_np_rio(bands, img_path))


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

# create a fastgs MSData object 
S2L1C_bands_55 = MSData.from_loader(
    S2L1C_desc,
    required_bands,
    [S2L1C_desc.rgb_combo["natural_color"],["B08"],["B10"]],
    get_ms_tensor_rio
)

test_fgs = FastGS.for_inference(S2L1C_bands_55,mask_codes)

# %%
# open the first image and check the stats
open_img = S2L1C_bands_55.load_image(tiles[0])
print(f'Shape {open_img.shape}\nMin val {open_img.min()}\nMax val {open_img.max()}')

# %%
open_img.show();

# %%
# build a fastai DataBlock
db = DataBlock(S2L1C_bands_55.create_xform_block())

# %%
# build a fastai dataloader
dl = db.dataloaders(source=tiles, bs=5)

# %%
learner = test_fgs.load_learner("/kaggle/working/models/S2_cloud_model", dl)

# %%
test_dl = learner.dls.test_dl(test_items=tiles, bs=1)

# %%
# #!zip -r file.zip /kaggle/working

# %%
#from IPython.display import FileLink
#FileLink(r'file.zip')

# %%
# show a batch, this does not work, not sure why?
test_dl.show_batch(max_n=5)

# %%
preds = learner.get_preds(dl=test_dl,with_decoded=True)[0]

# %%
preds.shape

# %%

# %%
