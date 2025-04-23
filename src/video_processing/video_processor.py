import os
import logging
import csv
import cv2 as cv
import numpy as np
from tqdm import tqdm

from utils.filename_builder import append_file_extension, create_out_filename
from utils.prep_cap import prep_cap
import utils.visualisers
from video_processing.optical_flow import analyse_sparse_optical_flow, calculate_angular_movement, estimate_rotation_center, estimate_rotation_center_individually

logger  = logging.getLogger(__name__)

class VideoProcessor:
    '''
    VideoProcessor class. This class is responsible for taking a video and performing user specified video processing on it. There are three options that the user can choose from: ['opt_flow', 'none', 'approx']. `opt_flow` performs optical flow on the video and then un-rotates it according to the data. `approx` will un-rotate based on the pre-calculated rotation that we got from the training data. `none` will just downscale the video and the appropriate sampling rate of the video.
    '''
    def __init__(self, method, config) -> None:
        assert method in ['opt_flow', 'none', 'approx'], 'method parameter must be one of: opt_flow, approx, none'

        # Due to some bugs with codecs, we are not able to keep the original 4k resolution
        # Therefore for downsampling this should be set to at least 2
        self.config        = config
        self.downscale_f   = self.config.get("downscale_factor")
        self.sampling_rate = self.config.get("sampling_rate")
        self.grayscale     = self.config.get("grayscale")
        self.start_at      = self.config.get("start_at")

        self.new_w   = None
        self.new_h   = None
        self.fps     = self.config.get("fps_out") // self.sampling_rate
        self.vid_in  = None
        self.vid_out = None
        self.cap_in  = None
        self.writer  = None
        self.total   = 0
        self.cap_w   = 0
        self.cap_h   = 0

        if method == 'opt_flow':
            self.method = self._optical_flow
        elif method == 'approx':
            self.method = self._guesstimate
        else:
            self.method = self._no_process


    def process_video(self, video_path, out_path) -> None:
        '''
        This is how the user should be interacting with this class. They specify the methods etc. when initialising the object and then process a video with `.process_video`
        '''
        self.vid_in  = video_path
        self.vid_out = out_path

        self.cap_in = prep_cap(video_path, self.start_at)

        self.fps = self.cap_in.get(cv.CAP_PROP_FPS)
        self.cap_w  = int(self.cap_in.get(cv.CAP_PROP_FRAME_WIDTH))
        self.cap_h  = int(self.cap_in.get(cv.CAP_PROP_FRAME_HEIGHT))

        self.new_w = int(self.cap_w // self.downscale_f)
        self.new_h = int(self.cap_h // self.downscale_f)

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        self.writer = cv.VideoWriter(out_path, fourcc, self.fps, (self.new_w, self.new_h))
        self.total = int(self.cap_in.get(cv.CAP_PROP_FRAME_COUNT)) - self.start_at

        self.method()


    def _optical_flow(self):
        '''
        Performs optical flow on the the video.
        '''
        f_params   = self.config.get("f_params")
        lk_params  = self.config.get("lk_params")
        num_points = self.config.get("num_points")

        np_trajectories = analyse_sparse_optical_flow(self.vid_in,
                                                      num_points=num_points,
                                                      lk_params=lk_params,
                                                      f_params=f_params,
                                                      start_at=self.start_at)

        center, quality = estimate_rotation_center_individually(np_trajectories)
        logger.info(f"Estimated rotation center: ({center[0]:.2f}, {center[1]:.2f})")
        logger.info(f"Center quality metric: {quality:.6f} (lower is better)")

        rotation_res = calculate_angular_movement(np_trajectories, center)

        base, _ = os.path.splitext(self.vid_in)

        graph_config = {
            'save_as': create_out_filename(base, [], ['of', 'analysis']),
            'save': True,
            'show': False
        }
        logger.info('Saving rotation analysis graph')
        utils.visualisers.visualize_rotation_analysis(np_trajectories, rotation_res, graph_config=graph_config)
        angles = rotation_res['average_angle_per_frame_deg']
        self._rotate_around_center(center, angles, 'optical flow')


    def _guesstimate(self):
        center_offset   = np.array((-59.06519626, -14.92924515))
        center_x        = self.cap_w // 2
        center_y        = self.cap_h // 2
        rotation_center = center_offset + (center_x, center_y)
        rot_per_frame   = np.float64(0.14798)
        logger.info('Rotation center: %s', rotation_center)
        logger.info('Rotation per frame: %s', rot_per_frame)

        angles = [0]
        angles.extend([rot_per_frame for i in range(self.total - 1)])

        self._rotate_around_center(rotation_center, angles, 'approximation')


    def _no_process(self):
        '''
        Performs basic video processing (subsampling, downscale etc..). In order to not copy code, it gets passed into _rotated_around_center with 0 angle rotation.
        '''
        center = (0,0)
        angles = [0 for i in range(self.total)]
        self._rotate_around_center(center, angles, 'no un-rotating')


    def _rotate_around_center(self, center, angles, name):
        '''
        Writes out the resulting video. Uses the configuration specified to also apply grayscale, sampling rate and downscale factor. 
        '''

        angle_idx = 0
        angle     = np.float64(0.0)

        logger.info('Correct number of angles: %s', len(angles) == self.total)

        transformations = []
        with tqdm(desc=f'Un-rotating with {name}', total=self.total) as pbar:
            i = 0
            while True:
                ret, frame = self.cap_in.read()

                if not ret or angle_idx >= len(angles):
                    break

                angle     += angles[angle_idx]
                angle_idx += 1

                if self.sampling_rate != 1 and i % self.sampling_rate != 0:
                    i += 1
                    pbar.update(1)
                    continue

                if self.grayscale:
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

                M             = cv.getRotationMatrix2D(center=center, angle=angle, scale=1)
                rotated_frame = cv.warpAffine(frame, M, (self.cap_w, self.cap_h))
                scaled_frame  = cv.resize(rotated_frame, (self.new_w, self.new_h))


                self.writer.write(scaled_frame)
                transformations.append(M)
                pbar.set_postfix(angle=f'{angle:.4f}')
                pbar.update(1)
                i +=1

        self.writer.release()
        self.cap_in.release()
        logger.info('Video saved in %s', self.vid_out)
        self._write_trans_into_csv(transformations) # Write transformations to a csv file


    def _write_trans_into_csv(self, transformations:list[np.ndarray]):
        base, _ = os.path.splitext(self.vid_out)

        save_as = create_out_filename(base, [], ['transformations'])
        save_as = append_file_extension(save_as, '.csv')

        with open(save_as, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_id','m00','m01','m02','m10','m11','m12'])

            for i, M in enumerate(transformations):
                writer.writerow([i] + M.ravel().tolist())

        logger.info('Stored csv data about transformations in %s', save_as)
