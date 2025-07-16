# Developer README

Welcome! This guide explains the internal structure and technical details of this codebase for high-resolution video registration of concrete samples.

## Project Overview

See the [README.md](/README.md) for the basic introduction.

### Pipeline stages

1. Data pre-processing: Splits the videos into two parts and cuts out the dark scenes. 
2. Rotation correction & video processing: Sparse optical flow for rotation correction and downsampling of the original videos. This step is fully configurable with the config file given in `src/video_processing/default_config.json5`. 
3. Video registration: There are three available modules for the video registration. `muDIC` that performs an area-based registration. `ORB` that performs feature-based registration. And `LightGlue` that performs video registration with the use a DL-based neural network. This step is fully configurable with the config file given in `src/video_registration/default_config.json5`. 
4. Image fusion: Creating composite images with the use of image fusion techniques. Implementation is in the `src/image_fusion/.
5. Evaluation: Various sharpness metrics. Implementation is in `src/image_evaluation/`

Each of the modules is well-documented and has a script that performs that specific step of the pipeline in `src/`. More information is available in the README given in `src/`.

## Dependencies

Below is a comprehensive list of the dependencies for this repository:

- [LightGlue](https://github.com/cvg/LightGlue) 
- [muDIC](https://mudic.readthedocs.io/en/latest/) 
- [OpenCV](https://opencv.org/) 
- [git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)
- [FFmpeg](https://ffmpeg.org/download.html) 
- [numpy](https://numpy.org/)

## Benchmarks

Are presented in the original thesis. However they can be recreated by registering all of the videos in `concrete-samples` with the default config files and then evaluating them with the `image_evaluation` module.

## Planned improvements

- [ ] Create a `main.py` file that will go through all of the steps of the pipeline.
- [ ] Better estimation of the registration precision.
- [ ] Registration of the front illumination.

## Maintainers

Erik Philippe Leclercq (Original author)
Research Centre Řež - Honza Blažek (Supervisor) - Imaging and Materials Lab
