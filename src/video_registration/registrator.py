import os 
import sys
import csv
import numpy as np
import cv2 as cv
import muDIC as dic

from tqdm import tqdm
from utils.filename_builder import append_file_extension, create_out_filename
from utils.prep_cap import prep_cap
import utils.visualisers

import logging

from video_registration.mudic_utils import correlate_matrix, create_mesh
from video_registration.video_matrix import create_video_matrix

logger = logging.getLogger(__name__)


class VideoRegistrator:
    def __init__(self, method, config) -> None:
        assert method in ['orb', 'lightglue', 'mudic'], 'method parameter must be one of: orb, lightglue, mudic'

        self.config = config

        self.vid_in  = None
        self.vid_out = None
        self.cap_in  = None

        self.frame_h = None
        self.frame_w = None

        if method == 'orb':
            self.method = self._orb

        elif method == 'mudic':
            self.method = self._mudic

        else:
            self.method = self._lightglue

    def register_video(self, vid_path, out_path):
        '''
        Registers a video into a numpy stack.
        '''
        self.vid_in  = vid_path
        self.vid_out = out_path

        self.cap_in = prep_cap(vid_path, set_to=0)

        self.frame_h = int(self.cap_in.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_w = int(self.cap_in.get(cv.CAP_PROP_FRAME_WIDTH))

        self.method()


    def _orb(self):
        print("TODO")


    def _mudic(self):
        vid_list = list(create_video_matrix(self.vid_in))
        image_stack = dic.image_stack_from_list(list(vid_list))
        mudic_config = self.config.get("mudic")
        mesh = create_mesh(self.frame_h,
                           self.frame_w,
                           image_stack,
                           mudic_config.get("box_h"),
                           mudic_config.get("box_w"),
                           mudic_config.get("num_elems_x"),
                           mudic_config.get("num_elems_y"))
        mesh_nodes = get_mesh_nodes(mesh)
        displacement = correlate_matrix(image_stack,
                                        mesh,
                                        self.config.get("ref_range"),
                                        self.config.get("max_it"))

        logger.info('mudic finished succesfully')



    def _lightglue(self):
        print("TODO")

    def _save_img_stack(self):
        print("TODO")
