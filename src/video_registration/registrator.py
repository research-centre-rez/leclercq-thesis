import logging
import os 
import sys
import csv
import numpy as np
import cv2 as cv
import muDIC as dic

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

        vid_stack   = create_video_matrix(self.vid_in, True)
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
        print("TODO")

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

