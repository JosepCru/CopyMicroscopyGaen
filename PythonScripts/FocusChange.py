from autoscript_tem_microscope_client import TemMicroscopeClient
from autoscript_tem_microscope_client.enumerations import *
from autoscript_tem_microscope_client.structures import *

microscope = TemMicroscopeClient()
microscope.connect()

initial_focus = microscope.optics.focus
print(initial_focus)

initial_defocus = microscope.optics.defocus
print(initial_defocus)

focus_step = 1E-6
steps = 10
focus_start = -focus_step * steps / 2

microscope.optics.reset_defocus()

for i in range(steps):
    defocus = i * focus_step + focus_start
    microscope.optics.defocus = defocus
