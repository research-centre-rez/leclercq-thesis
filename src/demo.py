# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: clean-venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Demo for Erik Leclercq's thesis
#
# This is a short demo that showcases the functionality of the pipeline that we implmented for this thesis. For more information you can look at the following:
#
# - `split_videos.py`: Video splitting with command-line interface
# - `process_video.py`: Performing rotation correction with optical flow and downsampling the video
# - `register_video.py`: Registering the downsampled scans
# - `fuse_video_stack.py`: Image fusion of the registered video stacks
# - `evalute_images.py`: Evaluation of the fused images

# %%
import os
import sys
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = "100000"

import logging
from utils import pprint

from video_processing import VideoProcessor, ProcessorMethod
from video_registration import VideoRegistrator, RegMethod
from image_fusion import ImageFuserFactory, FuseMethod, crop_image

from split_videos import _split_video
from utils import load_config, load_json_schema
import jsonschema
from jsonschema.exceptions import ValidationError

import matplotlib.pyplot as plt

from image_evaluation import NGLV, BrennerMethod
from evaluate_images import write_scores_to_csv

from image_evaluation.metrics import normType

# %%
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

logger = logging.getLogger(__name__)

# %% [markdown]
# ## Step 1: Split video
# Specify the path of the video we are interested in splitting and where we want to save the split videos.

# %%
video_path = '../data-backup/3A.MP4'
out_split_path = '../split-videos/'

# %%
_split_video(vid_path=video_path, out_path=out_split_path)

# %% [markdown]
# # Step 2: Downsampling and rotation correction

# %%
vid_processing_schema = load_json_schema('./video_processing/video_processing_schema.json')

# We select the video with the side angle illumination for the rotation correction
processing_input_filename = '../split-videos/3A/3A_part0.mp4'
out_filename = "../split-videos/3A/3A_part0_processed.mp4"

# Load the config and validate it against the json schema
try:
    video_processing_config = load_config('./video_processing/default_config.json5')
    jsonschema.validate(instance=video_processing_config, schema=vid_processing_schema)
    logger.info("Successfully loaded a JSON schema")
    pprint.pprint_dict(video_processing_config, desc="Config parameters")
except ValidationError as e:
    logger.error("Invalid configuration: \n %s", e.message)
    sys.exit(-1)

# Instantiate the video processor class
proc = VideoProcessor(method=ProcessorMethod.OPT_FLOW, config=video_processing_config)

# Extract the analysis of the motion
analysis = proc.get_rotation_analysis(processing_input_filename)

# Process the video with rotation correction and downsampling parameters.
proc.write_out_video(processing_input_filename, analysis, out_filename)

# %% [markdown]
# ## Step 3: Register the video
# There are three options for registration:
#
# - `RegMethod.MUDIC`: Performs MUDIC registration
# - `RegMethod.ORB`: Performs ORB registration
# - `RegMethod.LIGHTGLUE`: Performs LightGlue registration
#
# For LightGlue registration, a GPU with cuda capabilities is required.

# %%
reg_config_schema = load_json_schema('./video_registration/video_registration_schema.json')
reg_config_path = './video_registration/default_config.json5'

input_reg_path = out_filename
save_registered_block_as = '../split-videos/3A/3A_part0_registered'

try:
    reg_config = load_config(reg_config_path)
    jsonschema.validate(instance=reg_config, schema=reg_config_schema)
    logger.info("Successfully validated the submitted config")
    pprint.pprint_dict(reg_config, "Config parameters")
except jsonschema.ValidationError as e:
    logger.error("Invalid configuration: \n %s", e.message)
    sys.exit(1)

# Choosing a registration method
method = RegMethod.LIGHTGLUE

# Initialising the registrator class
reg = VideoRegistrator(method=method, config=reg_config)

reg_analysis = reg.get_registered_block(input_reg_path)

reg.save_registered_block(reg_analysis, save_registered_block_as)

# %% [markdown]
# ## Step 4: Image fusion

# %%
registered_stack_path = '../split-videos/3A/3A_part0_registered.npy'
save_fused_img_to = '../split-videos/3A/3A_part0_registered_fused.png'

fuse_types = [FuseMethod.MIN, FuseMethod.MAX, FuseMethod.MEAN]

fuser_factory = ImageFuserFactory()

# Use the min_fuser separately for cropping the fused images
min_fuser = fuser_factory.get_fuser(FuseMethod.MIN)

fusers = [fuser_factory.get_fuser(f_type) for f_type in fuse_types]

# Create a 'gallery' of fused images
gallery = {}
min_mask = min_fuser.get_min_mask(registered_stack_path)
for fuser in fusers:
    fused_image = fuser.get_fused_image(registered_stack_path)

    gallery[fuser.method] = crop_image(fused_image, min_mask)
    fuser.save_image_to_disc(gallery[fuser.method], save_fused_img_to)

# Display the gallery
fig, axes = plt.subplots(1, len(fusers), figsize=(15, 5))
for ax, fuser in zip(axes, fusers):
    method = fuser.method
    ax.imshow(gallery[method], cmap='gray')
    ax.set_title(str(method))
    ax.axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 5: Evalute the fused images
#
# The last step of our pipeline is to evaluate our fused images. We normalise the mean image so that the scores are all in the same range.

# %%


input_paths = [
    '../split-videos/3A/3A_part0_registered_fused_MAX.png',
    '../split-videos/3A/3A_part0_registered_fused_MIN.png',
    '../split-videos/3A/3A_part0_registered_fused_MEAN.png'
]
output_evaluation_path = '../split-videos/3A/evaluation.csv'

nglv_evaluator = NGLV()
brenner_evaluator = BrennerMethod()

normalisation = normType.l1_norm

scores = []
columns = ['Sample name', 'NGLV', 'Brenner']
for filename in input_paths:
    nglv_score = nglv_evaluator.calculate_metric(filename, normalise=True, normalisationType=normalisation)
    brenner_score = brenner_evaluator.calculate_metric(filename, normalise=True, normalisationType=normalisation)

    scores.append([filename, nglv_score, brenner_score])
    print(filename)
    print(f'  NGLV: {nglv_score:.4f},\n  Brenner: {brenner_score:.4f}')

write_scores_to_csv(output_evaluation_path, columns, scores)

