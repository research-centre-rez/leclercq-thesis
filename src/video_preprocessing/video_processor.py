import sys
import os
import logging
import cv2 as cv
import numpy as np
from tqdm import tqdm

from utils.filename_builder import create_out_filename
from utils.prep_cap import prep_cap
import utils.visualisers
from video_preprocessing.optical_flow import analyse_sparse_optical_flow, calculate_angular_movement, estimate_rotation_center, estimate_rotation_center_individually
class VideoProcessor:
    def __init__(self, sampling_rate:int, downscale_factor:int, gray_scale:bool, method:str, start_at) -> None:
        assert method in ['opt_flow', 'none', 'approx'], 'method parameter must be one of: opt_flow, approx, none'

        self.sampling_rate = max(1, sampling_rate)
        self.downscale_f = max(1, downscale_factor) # Due to some bugs with codecs, we are not able to keep the original 4k resolution
        self.grayscale = gray_scale
        self.start_at = start_at

        self.new_w   = None
        self.new_h   = None
        self.fps     = 25 // self.sampling_rate
        self.logger = logging.getLogger(__name__)
        self.vid_in  = None
        self.vid_out = None
        self.cap_in  = None
        self.writer  = None

        if method == 'opt_flow':
            self.method = self._optical_flow
        elif method == 'approx':
            self.method = self._guesstimate
        else:
            self.method = self._no_process
        

    def process_video(self, video_path, out_path):
        self.vid_in  = video_path
        self.vid_out = out_path

        self.cap_in = prep_cap(video_path, self.start_at)

        self.fps = self.cap_in.get(cv.CAP_PROP_FPS)
        frame_w  = int(self.cap_in.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_h  = int(self.cap_in.get(cv.CAP_PROP_FRAME_HEIGHT))

        self.new_w = int(frame_w // self.downscale_f)
        self.new_h = int(frame_h // self.downscale_f)

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        self.writer = cv.VideoWriter(out_path, fourcc, self.fps, (self.new_w, self.new_h))

        self.method()


    def _optical_flow(self):
        num_points = 15 # up for debate
        np_trajectories = analyse_sparse_optical_flow(self.vid_in, num_points)

        center, quality = estimate_rotation_center_individually(np_trajectories)
        self.logger.info(f"Estimated rotation center: ({center[0]:.2f}, {center[1]:.2f})")
        self.logger.info(f"Center quality metric: {quality:.6f} (lower is better)")

        rotation_res = calculate_angular_movement(np_trajectories, center)

        base, _ = os.path.splitext(self.vid_in)

        graph_config = {
            'save_as': create_out_filename(base, [], ['of', 'analysis']),
            'save': True,
            'show': False
        }
        self.logger.info('Saving rotation analysis graph')
        utils.visualisers.visualize_rotation_analysis(np_trajectories, rotation_res, graph_config=graph_config)
        angles = rotation_res['average_angle_per_frame_deg']
        self._rotate_around_center(center, angles, 'optical flow')


    def _guesstimate(self):
        frame_h = int(self.cap_in.get(cv.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(self.cap_in.get(cv.CAP_PROP_FRAME_WIDTH))
        
        center_offset = np.array((-59.06519626, -14.92924515))
        center_x = frame_w // 2
        center_y = frame_h // 2
        rotation_center = center_offset + (center_x, center_y)
        rot_per_frame   = np.float64(0.14798)
        self.logger.info('Rotation center: %s', rotation_center)
        self.logger.info('Rotation per frame: %s', rot_per_frame)

        total = int(self.cap_in.get(cv.CAP_PROP_FRAME_COUNT) - self.start_at)
        angles = [0]
        angles.extend([rot_per_frame for i in range(total - 1)])

        self._rotate_around_center(rotation_center, angles, 'approximation')


    def _no_process(self):
        '''
        Performs basic video processing (subsampling, downscale etc..)
        '''
        with tqdm(desc='Basic video processing', total=int(self.cap_in.get(cv.CAP_PROP_FRAME_COUNT)-self.start_at)) as pbar:
            i = 0
            while True:
                ret,frame = self.cap_in.read()
                if not ret:
                    break
                if self.sampling_rate != 1 and i % self.sampling_rate != 0:
                    i += 1
                    pbar.update(1)
                    continue

                new_frame = frame
                if self.grayscale:
                    # mp4 codecs don't support single channel videos. Therefore we have to
                    # do this weird conversion back and forth
                    new_frame = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
                    new_frame = cv.cvtColor(new_frame, cv.COLOR_GRAY2BGR)

                new_frame = cv.resize(new_frame, (self.new_w, self.new_h))
                self.writer.write(new_frame)

                pbar.set_postfix(i=f'{i}')
                i += 1
                pbar.update(1)

        self.writer.release()
        self.cap_in.release()
        cv.destroyAllWindows()


    def _rotate_around_center(self, center, angles, name):

        angle_idx = 0
        angle     = np.float64(0.0)

        total = int(self.cap_in.get(cv.CAP_PROP_FRAME_COUNT) - self.start_at)
        self.logger.info('Correct number of angles: %s', len(angles) == total)
        cap_w = int(self.cap_in.get(cv.CAP_PROP_FRAME_WIDTH))
        cap_h = int(self.cap_in.get(cv.CAP_PROP_FRAME_HEIGHT))

        with tqdm(desc=f'Un-rotating with {name}', total=total) as pbar:
            i = 0
            while True:
                ret, frame = self.cap_in.read()
                if not ret or angle_idx >= len(angles):
                    break

                angle += angles[angle_idx]
                angle_idx += 1

                if self.sampling_rate != 1 and i % self.sampling_rate != 0:
                    i += 1
                    pbar.update(1)
                    continue

                M             = cv.getRotationMatrix2D(center=center, angle=angle, scale=1)
                rotated_frame = cv.warpAffine(frame, M, (cap_w, cap_h))

                scaled_frame = cv.resize(rotated_frame, (self.new_w, self.new_h))


                self.writer.write(scaled_frame)

                pbar.set_postfix(angle=f'{angle:.4f}')
                i         +=1
                pbar.update(1)
        self.writer.release()
        self.cap_in.release()
        self.logger.info('Video saved')

