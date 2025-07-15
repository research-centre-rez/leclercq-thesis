# Structure of this directory

All of the code related to the thesis is placed here. The parts of the video processing pipeline are treated as python modules and each part has a corresponding script that can be used to run that part of the pipeline. The structure is the following:

- `data_preprocessing/`: Video splitting and removal of dark scenes. Can be used with `split_videos.py`.
- `video_processing/`: Downsampling of the pre-processed videos. Can be used with `video_processing.py`.
- `video_registration`: Registration of the processed scans. Can be used with `register_video.py`
- `image_fusion`: Fusion of the registered video stacks. Can be used with `fuse_video_stack.py`.
- `image_evaluation`: Evaluation of the fused images.

The core functionality of the code base is documented. The jupyter notebook `demo.ipynb` goes through the whole video processing pipeline step by step. If you wish to use the python scripts for video registration, please read their documentation by passing in the `--h` flag for each script.

For example, if you want to know how to use `register_video.py` you can read the manual with `python register_video.py --h`.
