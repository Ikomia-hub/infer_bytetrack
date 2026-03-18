# PyQt GUI framework
from PyQt6.QtWidgets import *

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion

from infer_bytetrack.infer_bytetrack_process import InferBytetrackParam


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferBytetrackWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferBytetrackParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        self.edit_categories = pyqtutils.append_edit(self.grid_layout, "Categories", self.parameters.categories)


        self.spin_conf_thres = pyqtutils.append_double_spin(
                                                self.grid_layout,
                                                "Confidence threshold",
                                                self.parameters.conf_thres,
                                                min=0., max=1.,
                                                step=0.01, decimals=2
        )

        self.spin_conf_thres_match = pyqtutils.append_double_spin(
                                                        self.grid_layout,
                                                        "Match confidence threshold",
                                                        self.parameters.conf_thres_match,
                                                        min=0., max=1.,
                                                        step=0.01, decimals=2
        )

        self.spin_track_buffer = pyqtutils.append_spin(
                                                    self.grid_layout,
                                                    "Track buffer",
                                                    self.parameters.track_buffer,
                                                    min=0., max=100
        )

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot
        self.parameters.conf_thres = self.spin_conf_thres.value()
        self.parameters.categories = self.edit_categories.text()
        self.parameters.conf_thres_match = self.spin_conf_thres_match.value()
        self.parameters.track_buffer = self.spin_track_buffer.value()
        self.parameters.update = True

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferBytetrackWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_bytetrack"

    def create(self, param):
        # Create widget object
        return InferBytetrackWidget(param, None)
