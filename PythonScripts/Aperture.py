from autoscript_tem_microscope_client import TemMicroscopeClient
from autoscript_tem_microscope_client.enumerations import *
from autoscript_tem_microscope_client.structures import *

microscope = TemMicroscopeClient()
microscope.connect()

C2 = microscope.optics.aperture_mechanisms.C2
aperture_start_position = C2.position
print(aperture_start_position)