from autoscript_tem_microscope_client import TemMicroscopeClient
from autoscript_tem_microscope_client.enumerations import *
from autoscript_tem_microscope_client.structures import *

microscope = TemMicroscopeClient()
microscope.connect()

image = microscope.acquisition.acquire_stem_image("HAADF", 512, 1E-6)
image.save(f"c:\\image.tiff")