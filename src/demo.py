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
from utils.visualisers import imshow


# %%
# Utility function for validating a JSON schema
def validate_json_config(config, json_schema):
    # Load the config and validate it against the json schema
    try:
        jsonschema.validate(instance=config, schema=json_schema)
        pprint.pprint_dict(config, desc="Config parameters")
    except ValidationError as e:
        sys.exit(-1)
    return config


# %% [markdown]
# ## Step 1: Split video
# Specify the path of the video we are interested in splitting and where we want to save the split videos.

# %%
# Relative path to the video you wish to process, as an example we take the following
video_path = os.path.join('..', 'concrete-samples', '3A.MP4')

# Where would you like to save the split videos? You only need to specify the output directory
out_split_path = os.path.join('..', 'split-videos', '')

# %%
_split_video(vid_path=video_path, out_path=out_split_path)

# %% [markdown]
# # Step 2: Downsampling and rotation correction

# %%
# Schema used for validating the input config file
video_processing_schema = load_json_schema(os.path.join('.', 'video_processing', 'video_processing_schema.json'))

# Video processing config
video_processing_config = load_config(os.path.join('.', 'video_processing', 'default_config.json5'))

# Name of the split video file you want to process
processing_input_filename = os.path.join('..', 'split-videos', '3A', '3A_part0.mp4')

# Specify how you want to name the processed video
# Note: name should not be the same as the input video
save_processed_video_as = os.path.join('..', 'split-videos', '3A', '3A_part0_processed.mp4')

# %%
validated_video_processing_config = validate_json_config(video_processing_config, video_processing_schema)
# Instantiate the video processor class
proc = VideoProcessor(method=ProcessorMethod.OPT_FLOW, config=validated_video_processing_config)

# Extract the analysis of the motion of the sample
analysis = proc.get_rotation_analysis(processing_input_filename)

# Write out the video video with rotation correction and downsampling parameters
proc.write_out_video(processing_input_filename, analysis, save_processed_video_as)

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
# Schema used for validating the input config file
video_registration_schema = load_json_schema(os.path.join('.', 'video_registration', 'video_registration_schema.json'))

# Config file used for video registration
video_registration_config = load_config(os.path.join('.', 'video_registration', 'default_config.json5'))

# Path to the video you want to register 
# Note: should be already pre-processed by step 2
video_registration_input = os.path.join('..', 'split-videos', '3A', '3A_part0_processed.mp4')

# How you want to save the registered block (numpy will automatically append the .npy file extension)
save_registered_block_as = os.path.join('..', 'split-videos', '3A', '3A_part0_registered')

# %%
validated_video_registration_config = validate_json_config(video_registration_config, video_registration_schema)

# Choosing a registration method
# Options are: LIGHTGLUE, ORB, MUDIC
# Note: MUDIC is quite slow with the registration
method = RegMethod.ORB

# Initialising the registrator class
reg = VideoRegistrator(method=method, config=validated_video_registration_config)

# Returns the registered video stack and the transformations that were used
reg_analysis = reg.get_registered_block(video_registration_input)

# Saves the registered block and the list of transformations to the specified location
reg.save_registered_block(reg_analysis, save_registered_block_as)

# %% [markdown]
# ## Step 4: Image fusion

# %%
# Path to the registered video stack
registered_stack_path = os.path.join('..', 'split-videos', '3A', '3A_part0_registered.npy')

# Specify where you want to save the fused images to
save_fused_images_to = os.path.join('..', 'split-videos', '3A', '3A_part0_fused.png')

# %%
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
    fuser.save_image_to_disc(gallery[fuser.method], save_fused_images_to)

imshow('Gallery of fused images', min_fusion=gallery[FuseMethod.MIN], max_fusion=gallery[FuseMethod.MAX], mean_fusion=gallery[FuseMethod.MEAN])

# %% [markdown]
# ## Step 5: Evalute the fused images
#
# The last step of our pipeline is to evaluate our fused images. We normalise the mean image so that the scores are all in the same range.

# %%
input_paths = [
    os.path.join('..', 'split-videos', '3A', '3A_part0_registered_fused_MAX.png'),
    os.path.join('..', 'split-videos', '3A', '3A_part0_registered_fused_MIN.png'),
    os.path.join('..', 'split-videos', '3A', '3A_part0_registered_fused_MEAN.png'),
]

output_evaluation_path = os.path.join('..', 'split-videos', '3A', 'evaluation.csv')

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

