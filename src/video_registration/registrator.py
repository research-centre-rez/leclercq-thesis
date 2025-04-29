import logging
import os 
import sys
import csv
import numpy as np
import cv2 as cv
import muDIC as dic
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd, numpy_image_to_torch

from skimage.measure import EllipseModel
from tqdm import tqdm
from utils.disp_utils import extract_medians
from utils.filename_builder import append_file_extension, create_out_filename
from utils.prep_cap import prep_cap

from utils.pprint import pprint_dict

from video_registration.mudic_utils import correlate_matrix, create_mesh, get_mesh_nodes
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
        mudic_config = self.config.get("mudic")
        pprint_dict(mudic_config, 'mudic in reg', logger)

        vid_stack   = create_video_matrix(self.vid_in, grayscale=True)
        image_stack = dic.image_stack_from_list(list(vid_stack))

        mesh = create_mesh(self.frame_h,
                           self.frame_w,
                           image_stack,
                           mudic_config.get("box_h"),
                           mudic_config.get("box_w"),
                           mudic_config.get("num_elems_x"),
                           mudic_config.get("num_elems_y"))

        displacement = correlate_matrix(image_stack,
                                        mesh,
                                        mudic_config.get("ref_range"),
                                        mudic_config.get("max_it"))
        displacement = displacement.squeeze()
        meds = extract_medians(displacement)

        logger.info('mudic finished succesfully')
        vid_stack = self._shift_by_vector(vid_stack, meds)

        np.save(self.vid_out, vid_stack)
        logger.info('Registered video saved to %s', self.vid_out)


    def _shift_by_vector(self, image_stack, displacement):
        n, h, w = image_stack.shape

        x_c, y_c = self._fit_ellipse(displacement)

        for i in tqdm(range(n), desc='Registering by shift'):
            image = image_stack[i]
            x_d, y_d = displacement[i]

            T = np.array([[1, 0, -x_d + x_c],
                          [0, 1, -y_d + y_c]])

            im_translated = cv.warpAffine(image, T, (w,h))
            image_stack[i] = im_translated

        return image_stack

    def _fit_ellipse(self, disp:np.ndarray):
        model = EllipseModel()
        success = model.estimate(disp)

        if not success:
            logger.error('failed to find an ellipse')
            sys.exit(-1) # TODO: Figure out what to do with this

        xc, yc, _, _, _ = model.params

        return xc,yc

    def _lightglue(self):
        _lglue_config = self.config.get('lightGlue')
        _hom_config   = _lglue_config['homography']
        _matcher       = _lglue_config['matcher']
        extractor     = _lglue_config['extractor']
        max_num_kp    = _lglue_config['max_num_keypoints']

        # Parsing the human-readable string into an opencv enum
        _hom_config['method'] = getattr(cv, _hom_config['method'])

        transformations = [np.eye(3,3)]

        vid_stack   = create_video_matrix(self.vid_in, grayscale=True)

        logger.info('Initialising the extractor and matcher')
        if extractor == "SuperPoint":
            extractor = SuperPoint(max_num_keypoints=max_num_kp).eval().cuda()
            matcher   = LightGlue(features='superpoint', **_matcher).eval().cuda()
        elif extractor == "DISK":
            extractor = DISK(max_num_keypoints=max_num_kp).eval().cuda()
            matcher   = LightGlue(features='disk', **_matcher).eval().cuda()
        elif extractor == "SIFT":
            extractor = SIFT(max_num_keypoints=max_num_kp).eval().cuda()
            matcher   = LightGlue(features='sift', **_matcher).eval().cuda()
        elif extractor == "ALIKED":
            extractor = ALIKED(max_num_kp=max_num_kp).eval().cuda()
            matcher   = LightGlue(features='aliked', **_matcher).eval().cuda()
        else:
            logger.error("Invalid keypoint extractor given, exiting.")
            sys.exit()

        fixed_np       = vid_stack[0]
        fixed_image = numpy_image_to_torch(cv.cvtColor(fixed_np, cv.COLOR_GRAY2RGB)).cuda()
        fixed_feats = extractor.extract(fixed_image)

        for i, moving in tqdm(enumerate(vid_stack[1:], start=1), total=len(vid_stack)-1, desc="Glueing"):
            moving_image = numpy_image_to_torch(cv.cvtColor(moving, cv.COLOR_GRAY2RGB)).cuda()
            moving_feats = extractor.extract(moving_image)
            matches      = matcher({'image0': fixed_feats, 'image1': moving_feats})

            # Removing bin dimension
            f_fixed, f_moved, matches01 = [rbd(x) for x in [fixed_feats, moving_feats, matches]]

            matches      = matches01['matches'] # indices with shape (K,2)
            points_fixed = f_fixed['keypoints'][matches[..., 0]]
            points_moved = f_moved['keypoints'][matches[..., 1]]

            H, _ = cv.findHomography(points_moved.cpu().numpy(), points_fixed.cpu().numpy(),
                                     **_hom_config)

            transformations.append(H)
            moving_warp = cv.warpPerspective(moving, H, (fixed_np.shape[1], fixed_np.shape[0]))
            vid_stack[i] = moving_warp
        self._write_transformation_into_csv(transformations)
        logger.info('Video successfully registered, saving as %s', self.vid_out)
        np.save(self.vid_out, vid_stack)


    def _save_img_stack(self):
        print("TODO")

    def _write_transformation_into_csv(self, trans:list[np.ndarray]):
        base, _ = os.path.splitext(self.vid_out)

        save_as = create_out_filename(base, [], ['transformations'])
        save_as = append_file_extension(save_as, '.csv')

        with open(save_as, 'w', newline='') as f:
            writer = csv.writer(f)

            for i, M in enumerate(trans):
                writer.writerow([i] + M.ravel().tolist())

        logger.info('Stored csv data about transformation in %s', save_as)

