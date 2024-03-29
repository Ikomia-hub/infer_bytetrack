# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
from ikomia import core, dataprocess

import numpy as np
from argparse import Namespace
from infer_bytetrack.yolox.tracker.byte_tracker import BYTETracker
from infer_bytetrack.utils import match_detections_with_tracks, xywh_xyxy


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferBytetrackParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.conf_thres = 0.25
        self.track_buffer = 30
        self.conf_thres_match = 0.7
        self.update = False
        self.categories = "all"

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.update = True
        self.conf_thres_match = float(param_map["conf_thres_match"])
        self.conf_thres = float(param_map["conf_thres"])
        self.track_buffer = int(param_map["track_buffer"])
        self.categories = str(param_map["categories"])

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {
            "conf_thres_match": str(self.conf_thres_match),
            "conf_thres": str(self.conf_thres),
            "track_buffer": str(self.track_buffer),
            "categories": str(self.categories)
        }
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferBytetrack(dataprocess.CObjectDetectionTask):

    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)
        # Add input/output of the process here
        self.remove_input(1)
        self.add_input(dataprocess.CObjectDetectionIO())
        self.add_input(dataprocess.CInstanceSegmentationIO())

        self.tracker = None

        # Create parameters class
        if param is None:
            self.set_param_object(InferBytetrackParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        self.categories = None

    def compute_color_for_labels(self, label):
        """
        Simple function that adds fixed color depending on the class
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in self.palette]
        return color

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()

        if self.tracker is None:
            args = Namespace()
            args.track_thresh = param.conf_thres
            args.track_buffer = param.track_buffer
            args.mot20 = False
            args.match_thresh = param.conf_thres_match
            self.tracker = BYTETracker(args)

        # Get input :
        task_input = self.get_input(0)

        # Get image from input/output (numpy array):
        src_image = task_input.get_image()
        img_size = np.shape(src_image)

        # Get object input
        dets = self.get_input(1).get_objects()
        inst_segs = self.get_input(2).get_objects()

        # Get label input
        self.categories = param.categories
        while ', ' in self.categories:
            self.categories = self.categories.replace(", ", ",")
        labels_to_track = self.categories.split(',')

        # Tracking for object detection input
        if len(dets):
            self.set_output(dataprocess.CObjectDetectionIO(), 1)
            # Get output :
            task_output = self.get_output(1)
            task_output.init("ByteTrack", 0)
            tracks = self.tracker.update(
                                    np.array([xywh_xyxy(o.box) + [o.confidence] for o in dets]),
                                    img_size,
                                    img_size
            )
            if len(tracks) > 0:
                pairings = match_detections_with_tracks(dets, tracks)
                for k, v in pairings.items():
                    det = dets[k]
                    if param.categories == "all" or det.label in labels_to_track:
                        color = self.compute_color_for_labels(v)
                        task_output.add_object(v, det.label, det.confidence, *det.box, color)

        # Tracking for instance segmentation input
        elif len(inst_segs):
            self.set_output(dataprocess.CInstanceSegmentationIO(), 1)
            # Get output :
            task_output = self.get_output(1)
            task_output.init("ByteTrack", 0, img_size[1], img_size[0])
            tracks = self.tracker.update(
                                    np.array([xywh_xyxy(o.box) + [o.confidence] for o in inst_segs]),
                                    img_size,
                                    img_size
            )
            if len(tracks) > 0:
                pairings = match_detections_with_tracks(inst_segs, tracks)
                for k, v in pairings.items():
                    inst_seg = inst_segs[k]
                    if param.categories == "all" or det.label in labels_to_track:
                        color = self.compute_color_for_labels(v)
                        task_output.add_object(
                                            v,
                                            0,
                                            0,
                                            inst_seg.label,
                                            inst_seg.confidence,
                                            *inst_seg.box,
                                            inst_seg.mask,
                                            color
                        )

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferBytetrackFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_bytetrack"
        self.info.short_description = "Infer ByteTrack for object tracking"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Tracking"
        self.info.version = "1.0.1"
        self.info.icon_path = "icons/logo.png"
        self.info.authors = "Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, "\
                            "Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, " \
                            "Ping and Liu, Wenyu and Wang, Xinggang"
        self.info.article = "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
        self.info.journal = "Proceedings of the European Conference on Computer Vision (ECCV)"
        self.info.year = 2022
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/abs/2110.06864"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_bytetrack"
        self.info.original_repository = "https://github.com/ifzhang/ByteTrack"
        # Keywords used for search
        self.info.keywords = "multiple, object, tracking, kalman"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OBJECT_TRACKING"

    def create(self, param=None):
        # Create process object
        return InferBytetrack(self.info.name, param)
