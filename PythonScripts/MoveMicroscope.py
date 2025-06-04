from autoscript_tem_microscope_client import TemMicroscopeClient
from autoscript_tem_microscope_client.enumerations import *
from autoscript_tem_microscope_client.structures import *

microscope = TemMicroscopeClient()
microscope.connect()

InitialStagePosition = microscope.stage.position
fov = microscope.optics.scan_field_of_view

Overlap = 0.1
shift = fov * (1 - Overlap)

shiftX = 2 * shift
shiftY = 1 * shift

position = InitialStagePosition + StagePosition(x=shiftY, y=shiftX)
microscope.stage.absolute_move(position)
print("Microscope Moved to" + position)