from autoscript_tem_microscope_client import TemMicroscopeClient
from autoscript_tem_microscope_client.enumerations import *
from autoscript_tem_microscope_client.structures import *
from matplotlib import pyplot as plot

print("Starting example script...")

# Connect to AutoScript server
microscope = TemMicroscopeClient()
microscope.connect()

# Change optical mode to STEM
microscope.optics.optical_mode = OpticalMode.STEM

# Read out microscope type and software version
system_info = microscope.service.system.name + ", version " + microscope.service.system.version
print("System info: " + system_info)

# Read out available detectors and cameras
camera_detectors = microscope.detectors.camera_detectors
print("Camera detectors: ", camera_detectors)
scanning_detectors = microscope.detectors.scanning_detectors
print("Scanning detectors: ", scanning_detectors)

# Grab one image with HAADF detector if present
if DetectorType.HAADF in scanning_detectors:
    print("Taking image with HAADF detector...")
    image = microscope.acquisition.acquire_stem_image(DetectorType.HAADF, 1024, 50e-9)

    # Show the image on a popup window
    plot.imshow(image.data, cmap="gray")
    plot.show()

print("Example script finished successfully.")

microscope.disconnect()