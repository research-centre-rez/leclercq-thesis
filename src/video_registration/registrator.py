from concurrent.futures import ProcessPoolExecutor
from enum import Enum
import time
import logging
import os
import sys
import csv
import numpy as np
import cv2 as cv
import muDIC as dic
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd, numpy_image_to_torch
import torch
import torch.nn.functional as F

from skimage.measure import EllipseModel
from tqdm import tqdm
from utils.disp_utils import extract_medians
from utils.filename_builder import append_file_extension, create_out_filename
from utils.prep_cap import prep_cap

from utils.pprint import pprint_dict

from video_registration.mt_homography import compute_and_apply_homography
from video_registration.mudic_utils import correlate_matrix, create_mesh, get_mesh_nodes
from video_registration.video_matrix import create_video_matrix

logger = logging.getLogger(__name__)


class RegMethod(Enum):
    ORB = 0
    MUDIC = 1
    LIGHTGLUE = 2


class VideoRegistrator:
    def __init__(self, method: RegMethod, config: dict) -> None:

        self.config = config

        self.vid_in = None
        self.vid_out = None
        self.cap_in = None

        self.frame_h = None
        self.frame_w = None

        if method == RegMethod.ORB:
            self.method = self._get_orb_registration

        elif method == RegMethod.MUDIC:
            self.method = self._get_mudic_registration

        elif method == RegMethod.LIGHTGLUE:
            self.method = self._get_lightglue_registration

    def save_registered_block(self, reg_analysis: dict, save_as: str) -> None:
        """
        Write the registered block and the transformations to memory.
        """

        np.save(save_as, reg_analysis["registered_block"])
        # self._write_transformation_into_csv(reg_analysis["transformations"], save_as)

    def get_registered_block(self, video_input_path: str) -> dict:
        """
        Performs video registration based on the user's specified method.

        Args:
            video_input_path (str): path to the video you want to register

        Returns:
            dictionary that contains two fields:
                "registered_block": registered block
                "transformations": transformations that were performed at each step
        """
        reg_block, transformations = self.method(video_input_path)
        result = {"registered_block": reg_block, "transformations": transformations}
        return result

    def _get_lightglue_registration(
        self, input_video: str
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Performs LightGlue registration
        """
        _lglue_config = self.config.get("lightGlue")
        _hom_config = _lglue_config["homography"]
        _matcher = _lglue_config["matcher"]
        extractor = _lglue_config["extractor"]
        max_num_kp = _lglue_config["max_num_keypoints"]

        extractor, matcher = self._get_extractor_matcher(
            extractor_name=extractor, matcher_config=_matcher, max_num_kp=max_num_kp
        )

        # Parsing the human-readable string into an opencv enum
        _hom_config["method"] = getattr(cv, _hom_config["method"])

        # First image in the sequence has no transformation
        transformations = [(0, np.eye(3, 3))]

        vid_stack = create_video_matrix(input_video, grayscale=True)

        logger.info("Initialising the extractor and matcher")

        fixed_np = vid_stack[0]

        fixed_image = numpy_image_to_torch(
            cv.cvtColor(fixed_np, cv.COLOR_GRAY2RGB)
        ).cuda()
        fixed_feats = extractor.extract(fixed_image)

        N = 4
        f_kp = fixed_feats["keypoints"]
        f_dp = fixed_feats["descriptors"]
        im_size = fixed_feats["image_size"]

        f_kp = f_kp.repeat((N, 1, 1))
        f_dp = f_dp.repeat((N, 1, 1))
        im_size = im_size.repeat((N, 1))

        fixed_feats = {"keypoints": f_kp, "descriptors": f_dp, "image_size": im_size}

        out_res = (fixed_np.shape[1], fixed_np.shape[0])

        futures = []
        with torch.no_grad(), ProcessPoolExecutor(max_workers=16) as executor:
            for i in tqdm(
                range(1, len(vid_stack[1:]), N),
                desc="Glueing",
            ):
                moving_images = []
                for j in range(N):
                    try:
                        moving_images.append(
                            numpy_image_to_torch(
                                cv.cvtColor(vid_stack[j + i], cv.COLOR_GRAY2RGB)
                            ).cuda()
                        )
                    except IndexError as e:
                        moving_images.append(moving_images[0].cuda())

                moving_images = torch.stack(moving_images).cuda()
                moving_feats = extractor.extract(moving_images)

                # This is staying here until I am sure I can safely delete it away
                # batched_kp = []
                # batched_dp = []
                # batched_imsize = []
                # max_kp = 0
                # for j in range(N):
                #     try:
                #         moving_image = numpy_image_to_torch(
                #             cv.cvtColor(vid_stack[j + i], cv.COLOR_GRAY2RGB)
                #         ).cuda()
                #     except IndexError as e:
                #         moving_image = torch.zeros_like(fixed_image).cuda()
                #     moving_feats = extractor.extract(moving_image)
                #     v_kp = moving_feats["keypoints"]
                #     v_dp = moving_feats["descriptors"]
                #     im_size = moving_feats["image_size"]
                #     max_kp = max(max_kp, v_kp.shape[1])
                #     batched_kp.append(v_kp)
                #     batched_dp.append(v_dp)
                #     batched_imsize.append(im_size)

                # padded_kp = [
                #     F.pad(kp, (0, 0, 0, max_kp - kp.shape[1])) for kp in batched_kp
                # ]
                # padded_dp = [
                #     F.pad(kp, (0, 0, 0, max_kp - kp.shape[1])) for kp in batched_dp
                # ]
                # stacked_kp = torch.cat(padded_kp)
                # stacked_dp = torch.cat(padded_dp)

                # moving_feats = {
                #     "keypoints": stacked_kp,
                #     "descriptors": stacked_dp,
                #     "image_size": torch.cat(batched_imsize),
                # }
                matches = matcher({"image0": fixed_feats, "image1": moving_feats})

                cpu_fixed_kps = fixed_feats["keypoints"].cpu().numpy()
                cpu_moving_kps = moving_feats["keypoints"].cpu().numpy()

                for idx, match in enumerate(matches["matches"]):
                    if idx + i < len(vid_stack):
                        futures.append(
                            executor.submit(
                                compute_and_apply_homography,
                                cpu_fixed_kps[idx],
                                cpu_moving_kps[idx],
                                match.cpu().numpy(),
                                vid_stack[idx + i],
                                _hom_config,
                                out_res,
                                idx + i,
                            )
                        )
                    else:
                        break

        for future in futures:
            result = future.result()
            if result is not None:
                frame_idx, H, warped = result
                transformations.append((frame_idx, H))
                vid_stack[frame_idx] = warped

        return vid_stack, transformations

    #                for idx, match in enumerate(matches["matches"]):
    #                    if len(match) > 4:
    #                        points_fixed = fixed_feats["keypoints"][idx][match[...,0]]
    #                        points_moved = moving_feats["keypoints"][idx][match[...,1]]
    #                        H, _ = cv.findHomography(
    #                            points_moved.cpu().numpy(),
    #                            points_fixed.cpu().numpy(),
    #                            **_hom_config
    #                        )
    #
    #                        transformations.append(H)
    #                        moving_warp = cv.warpPerspective(
    #                            vid_stack[i+idx], H, (fixed_np.shape[1], fixed_np.shape[0])
    #                        )
    #
    #                        vid_stack[i+idx] = moving_warp
    #
    #            return vid_stack, transformations

    #                moving_image = numpy_image_to_torch(
    #                    cv.cvtColor(moving, cv.COLOR_GRAY2RGB)
    #                ).cuda()
    #                moving_feats = extractor.extract(moving_image)
    #
    #                # Removing batch dimension
    #                f_fixed, f_moved, matches01 = [
    #                    rbd(x) for x in [fixed_feats, moving_feats, matches]
    #                ]
    #
    #                matches = matches01["matches"]  # indices with shape (K,2)
    #                points_fixed = f_fixed["keypoints"][matches[..., 0]]
    #                points_moved = f_moved["keypoints"][matches[..., 1]]
    #
    #                H, _ = cv.findHomography(
    #                    points_moved.cpu().numpy(),
    #                    points_fixed.cpu().numpy(),
    #                    **_hom_config
    #                )
    #
    #                transformations.append(H)
    #                moving_warp = cv.warpPerspective(
    #                    moving, H, (fixed_np.shape[1], fixed_np.shape[0])
    #                )
    #
    #                vid_stack[i] = moving_warp
    #
    #            logger.info("Video succesfully registered.")
    #            return vid_stack, transformations

    def _get_mudic_registration(
        self, input_video
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Performs video registration with the use of muDIC.
        """

        mudic_config = self.config.get("mudic")

        vid_stack = create_video_matrix(input_video, grayscale=True)
        image_stack = dic.image_stack_from_list(list(vid_stack))

        frame_h, frame_w = vid_stack.shape[1:]

        mesh = create_mesh(
            frame_h, frame_w, image_stack, **mudic_config.get("mesh_parameters")
        )

        displacement = correlate_matrix(
            image_stack, mesh, mudic_config.get("ref_range"), mudic_config.get("max_it")
        )
        displacement = displacement.squeeze()
        meds = extract_medians(displacement)

        logger.info("mudic finished succesfully")
        return self._shift_by_vector(vid_stack, meds)

    def _get_ORB_registration(self, input_video) -> tuple[np.ndarray, list[np.ndarray]]:
        return None

    def _shift_by_vector(
        self, image_stack, displacement
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        n, h, w = image_stack.shape

        x_c, y_c = self._fit_ellipse(displacement)
        transformations = [np.eye(2, 3)]

        for i in tqdm(range(n), desc="Registering by shift"):
            image = image_stack[i]
            x_d, y_d = displacement[i]

            T = np.array([[1, 0, -x_d + x_c], [0, 1, -y_d + y_c]])
            transformations.append(T)

            im_translated = cv.warpAffine(image, T, (w, h))
            image_stack[i] = im_translated

        return image_stack, transformations

    def _fit_ellipse(self, disp: np.ndarray):
        model = EllipseModel()
        success = model.estimate(disp)

        if not success:
            logger.error("failed to find an ellipse")
            sys.exit(-1)  # TODO: Figure out what to do with this

        xc, yc, _, _, _ = model.params

        return xc, yc

    def _write_transformation_into_csv(self, trans: list[np.ndarray], save_as: str):
        base, _ = os.path.splitext(save_as)

        save_as = create_out_filename(base, [], ["transformations"])
        save_as = append_file_extension(save_as, ".csv")

        with open(save_as, "w", newline="") as f:
            writer = csv.writer(f)

            for i, M in enumerate(trans):
                writer.writerow([i] + M.ravel().tolist())

        logger.info("Stored csv data about transformation in %s", save_as)

    def _get_extractor_matcher(
        self, extractor_name: str, matcher_config: dict, max_num_kp
    ):
        if extractor_name == "SuperPoint":
            extractor = SuperPoint(max_num_keypoints=max_num_kp).eval().cuda()
            matcher = LightGlue(features="superpoint", **matcher_config).eval().cuda()
        elif extractor_name == "DISK":
            extractor = DISK(max_num_keypoints=max_num_kp).eval().cuda()
            matcher = LightGlue(features="disk", **matcher_config).eval().cuda()
        elif extractor_name == "SIFT":
            extractor = SIFT(max_num_keypoints=max_num_kp).eval().cuda()
            matcher = LightGlue(features="sift", **matcher_config).eval().cuda()
        elif extractor_name == "ALIKED":
            extractor = ALIKED(max_num_kp=max_num_kp).eval().cuda()
            matcher = LightGlue(features="aliked", **matcher_config).eval().cuda()
        elif extractor_name == "DoGHardNet":
            extractor = DoGHardNet(max_num_kp=max_num_kp).eval().cuda()
            matcher = LightGlue(features="doghardnet", **matcher_config).eval().cuda()
        else:
            logger.error("Invalid keypoint extractor given, exiting.")
            sys.exit()

        return extractor, matcher
