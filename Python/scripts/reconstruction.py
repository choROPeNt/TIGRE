import os 
import tigre
import tigre.algorithms as algs
from tigre.utilities import gpu
import yaml
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_config():
    parser = argparse.ArgumentParser(description='Reconstruction')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    return config



def load_radios(path,dtype=".tif"):
    radios_list = []
    for n, file in enumerate(sorted(os.listdir(path))):
        if file.endswith(dtype):
            file_path = os.path.join(path, file)
            img = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)
            print(f"Loaded: {file} with dtype {img.dtype}")
            radios_list.append(img.astype(np.float32))

    if radios_list:
        radios = np.concatenate([np.expand_dims(img, axis=0) for img in radios_list], axis=0)
        print(f"Shape of concatenated projections: {radios.shape}")
    else:
        radios = None
        print("No images were loaded.")    
    return radios


def main():
    # Load configuration
    config = load_config()

    targetGpuName = 'NVIDIA A100-SXM4-40GB'  # noqa: N816
    # You can get the list of GPU IDs
    gpuids = gpu.getGpuIds(targetGpuName)
    print("Result of ({})".format(targetGpuName))
    print("\t Number of devices: {}".format(len(gpuids)))
    print("\t gpuids: {}".format(gpuids))


    radios = load_radios(config["radios"])

        # Extracting CERA data
    cera_data = {
        "source_object_distance": 153.878571,
        "source_image_distance": 947.836914,
        "detector_offset_u": 1598.5,
        "detector_offset_v": 1149.5,
        "pixel_size_u": 0.127,
        "pixel_size_v": 0.127,
        "num_channels_per_row": 2302,
        "num_rows": 3198,
        "start_angle": 0.006056,
        "scan_angle": 360.0
    }

    # Geometry setup
    geo = tigre.geometry_default()
    
    geo.DSD = cera_data["source_image_distance"]  # Distance Source Detector
    geo.DSO = cera_data["source_object_distance"]  # Distance Source Origin

    geo.nDetector = np.array([cera_data["num_channels_per_row"], cera_data["num_rows"]])  # number of pixels (U, V)
    geo.dDetector = np.array([cera_data["pixel_size_u"], cera_data["pixel_size_v"]])  # size of each pixel (U, V)
    geo.sDetector = geo.nDetector * geo.dDetector  # total size of the detector (U, V)
    
    
    geo.offDetector = np.array([cera_data["detector_offset_v"], cera_data["detector_offset_u"]]) * geo.dDetector  # Offset of detector

    # Setting the number of projections and angles
    num_radios = radios.shape[0]
    start_angle = cera_data["start_angle"]
    scan_angle = cera_data["scan_angle"]
    angles = np.linspace(start_angle, start_angle + np.deg2rad(scan_angle), num_radios)


    #%% Reconstruction

    # FDK
    imgFDK = algs.fdk(radios, geo, angles,gpuids=gpuids)

    print(f"Shape of reconstruction {imgFDK.shape}")
    plt.imshow(imgFDK[512//2,::])
    plt.savefig("test.png")





if __name__ == "__main__":
    main()