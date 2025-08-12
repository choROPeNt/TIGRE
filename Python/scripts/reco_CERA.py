import os
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import configparser
## TIGRE 
import tigre
import tigre.algorithms as algs
from tigre.utilities import gpu

from skimage.restoration import denoise_tv_chambolle

from contextlib import contextmanager

@contextmanager
def time_it(label="Code block"):
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{label} took {elapsed_time:.4f} seconds to execute.")


def read_h5(path,key="sinogram"):

    with h5py.File(path,"r") as h5:
        data = h5[key][:].astype(np.float32)

    return data

def read_config_as_dict(config_path):
    # Create a config parser object with no interpolation to avoid errors with '%'
    config = configparser.ConfigParser(interpolation=None)
    
    # Preserve case sensitivity of keys
    config.optionxform = str

    # Read the config file
    config.read(config_path)

    # Create a dictionary to store all sections and keys
    config_dict = {}

    # Iterate over each section in the config
    for section in config.sections():
        # Store the section as a dictionary of key-value pairs
        config_dict[section] = dict(config.items(section))
    
    return config_dict


def main():
    ######################################################
    path = "/data/horse/ws/dchristi-TIGRE/Calibr_Z200_2023-06-15__00"
    result_path = "results"
    scan_name = os.path.basename(path)
    sinogram_file = "Calibr_Z200_2023-06-15__00.sino.corr.gzip.h5"
    config_file = "Calibr_Z200_2023-06-15_.config"

    algorithm = "FDK"

    default_cmap = "plasma_r"
    ######################################################
    
    if os.path.exists(os.path.join(path,config_file)):
        print("#===========data-loading=================")
        print(f"loading config file {config_file} in {path}")
        config_dict = read_config_as_dict(os.path.join(path,config_file))

        print(config_dict)


        
        print("#===========data-loading=================")
        print(f"loading sinogram file {sinogram_file} in {path}")
        with time_it("Reading H5 file"):
            data = read_h5(os.path.join(path,sinogram_file))
        
        
        # print(f"data loaded in {t1-t0} seconds")
        print(f"shape of sinogram: {data.shape}")
        print(f"datatype of sinogram: {data.dtype}")
        print(f"min of sinogram: {data.min()}, max of sinogram: {data.max()}")
        print(f"memory {data.nbytes/1024**2:.3f} Mbytes")

        assert (
            config_dict["CustomKeys"]["NumProjections"] == data.shape[0]
            ), "NumProjections do not match"

        NumProjections = int(config_dict["CustomKeys"]["NumProjections"])
        #TODO think about wheter start angle is in degrees or radians
        theta_start = float(config_dict["CustomKeys.ProjectionMatrices"]["StartAngle"])
        theta_end = float(config_dict["CustomKeys.ProjectionMatrices"]["ScanAngle"])

        # print(theta_start,np.radians(theta_end),NumProjections)
        data = -np.log(data / 2800)

        theta = np.linspace(np.radians(theta_start),np.radians(theta_end),NumProjections)

        fig , axs = plt.subplots(1,2,figsize=(2*4.8,4.8))
        axs = axs.flatten()
        cax = axs[0].imshow(data[:, data.shape[1]//2, :], 
                            cmap=default_cmap, 
                            aspect='auto',
                            norm=LogNorm()
                            )
        
        axs[0].set_title(f"Sinogram ($v=${data.shape[1]//2} [Px])")
        axs[0].set_xlabel(r"$u$ [Px]")
        axs[0].set_ylabel(r"$\theta$ [rad]")
        
        # Add a colorbar to axs[0]
        cbar = fig.colorbar(cax, ax=axs[0], orientation='vertical')
        cbar.set_label(r'$\log{(I)}$ [16bit]')
        # Replace y-axis with theta values
        num_ticks = 10
        tick_positions = np.linspace(0, data.shape[0] - 1, num_ticks).astype(int)
        tick_labels = [f"{theta[i]:.2f}" for i in tick_positions]
        axs[0].set_yticks(tick_positions)
        axs[0].set_yticklabels(tick_labels)

        cax = axs[1].imshow(data[:, :,data.shape[2]//2], 
                            cmap=default_cmap, 
                            aspect='auto',
                            norm=LogNorm()
                            )
        
        axs[1].set_title(f"Sinogram ($u=${data.shape[2]//2} [Px])")
        axs[1].set_xlabel(r"$v$ [Px]")
        axs[1].set_ylabel(r"$\theta$ [rad]")
        
        # Add a colorbar to axs[0]
        cbar = fig.colorbar(cax, ax=axs[1], orientation='vertical')
        cbar.set_label(r'$\log{(I)}$ [16bit]')
        # Replace y-axis with theta values
        num_ticks = 10
        tick_positions = np.linspace(0, data.shape[0] - 1, num_ticks).astype(int)
        tick_labels = [f"{theta[i]:.2f}" for i in tick_positions]
        axs[1].set_yticks(tick_positions)
        axs[1].set_yticklabels(tick_labels)


        fig.tight_layout()
        fig.savefig(os.path.join(result_path, scan_name + ".sino.pdf"),dpi=300)

        #TODO make it a config
        targetGpuName = 'NVIDIA A100-SXM4-40GB'  # noqa: N816  
        # You can get the list of GPU IDs
        gpuids = gpu.getGpuIds(targetGpuName)
        print("#===========GPUs=========================")
        print("Result of ({})".format(targetGpuName))
        print("\t Number of devices: {}".format(len(gpuids)))
        print("\t gpuids: {}".format(gpuids))


        print("#===========Geometrie=========================")
        print(f"setting up cone beam geometrie from {config_file} ")

        # Geometry setup
        geo = tigre.geometry_default()
        
        ## Geo
        geo.DSD = float(config_dict["CustomKeys.ProjectionMatrices"]["SourceImageDistance"])  # Distance Source Detector
        geo.DSO = float(config_dict["CustomKeys.ProjectionMatrices"]["SourceObjectDistance"])  # Distance Source Origin
        
        ## Detector 
        geo.nDetector = np.array([data.shape[1], data.shape[2]])  # number of pixels (U, V)
        geo.dDetector = np.array([float(config_dict["Projections"]["PixelSizeU"]), 
            float(config_dict["Projections"]["PixelSizeV"])])  # size of each pixel (U, V)
        geo.sDetector = geo.nDetector * geo.dDetector  # total size of the detector (U, V)
    
        ## Volume:
        SizeX = float(config_dict["Volume"]["SizeX"])
        SizeY = float(config_dict["Volume"]["SizeY"])
        SizeZ = float(config_dict["Volume"]["SizeZ"])
        VoxelSizeX = float(config_dict["Volume"]["VoxelSizeX"])
        VoxelSizeY = float(config_dict["Volume"]["VoxelSizeY"])
        VoxelSizeZ = float(config_dict["Volume"]["VoxelSizeZ"])
        
        ##TODO remove for production
        scale = 1

        geo.nVoxel = np.array([SizeZ, SizeX,SizeY])//scale
        geo.dVoxel = np.array([VoxelSizeZ,VoxelSizeX , VoxelSizeY])*scale
        geo.sVoxel = geo.nVoxel * geo.dVoxel


        geo.offDetector = np.array([0,0]) ##TODO is the offset per deauflt in the center?
        geo.accuracy = 0.5
        geo.mode = "cone"

        ##TODO implementation per angle
        # geo.COR = -13.611205805 * geo.dDetector[0] ## from cera software meanu can be adjusted to per angle in [mm]
        ##TODO output of figure
        tigre.plot_geometry(geo, angle=-np.pi / 6)
    
        print("#===========Reconstruction=========================")
    
        # FDK
        print(f"reconstructing {sinogram_file} with shape {data.shape}")
        print(f"output shape of volume: {geo.nVoxel} [Px]")
        print(f" size of volume: {geo.sVoxel} [mm]")
        print(f"voxel  size of volume: {geo.dVoxel} [mm/Px]")

        with time_it("Reconstructing with FDK"):
            vol = algs.fdk(data,             # sinogram
                              geo,              # geometry file
                              theta,            # angels in radians
                              gpuids=gpuids,    # list of gpus
                              filter="cosine"   # filter options: 'ram_lak' (default),'shepp_logan','cosine','hamming','hann'
                              )

        min_val = vol.min()
        max_val = vol.max()

        # Step 2: Rescale the data to the range of int16 [-32768, 32767]
        # (You can adjust the target range if necessary for your specific use case)
        vol = (vol - min_val) / (max_val - min_val)  # Normalize to [0, 1]
        vol = vol * (2**15 - 1)  # Scale to the range [0, 32767]
        vol = vol.astype(np.int16)  # Convert to int16
        
        # vol = vol.astype(np.int16)
        print(f"shape of volume: {vol.shape}")
        print(f"datatype of sinogram: {vol.dtype}")
        print(f"min of volume: {vol.min()}, max of volume: {vol.max()}")
        print(f"memory size {vol.nbytes/1024**2:.3f} Mbytes")


        fig, axs = plt.subplots(2,2, figsize=(2*4.8,2*4.8))
        axs = axs.flatten()
        cax = axs[0].imshow(vol[:,:,vol.shape[2]//2],
                            cmap=default_cmap,
                            aspect='auto',
                            # norm=LogNorm()
                            )
        
        axs[0].set_title(f"Volume ($y=${vol.shape[2]//2} [Px])")
        axs[0].set_xlabel(r"$x$ [Px]")
        axs[0].set_ylabel(r"$z$ [Px]")
        
        # Add a colorbar to axs[0]
        cbar = fig.colorbar(cax, ax=axs[0], orientation='vertical')
        cbar.set_label('Intensity [16bit]')

        cax = axs[1].imshow(vol[:,vol.shape[1]//2,:],
                            cmap=default_cmap,
                            aspect='auto',
                            # norm=LogNorm()
                            )

        axs[1].set_title(f"Volume ($x=${vol.shape[1]//2} [Px])")
        axs[1].set_xlabel(r"$y$ [Px]")
        axs[1].set_ylabel(r"$z$ [Px]")
        #    Add a colorbar to axs[1]
        cbar = fig.colorbar(cax, ax=axs[1], orientation='vertical')
        cbar.set_label('Intensity [16bit]')


        axs[2].imshow(vol[vol.shape[0]//2,:,:],
                      cmap=default_cmap,
                      aspect='auto',
                    #   norm=LogNorm()
                      )
        
        axs[2].set_title(f"Volume ($z=${vol.shape[0]//2} [Px])")
        axs[2].set_xlabel(r"$x$ [Px]")
        axs[2].set_ylabel(r"$y$ [Px]")
        #    Add a colorbar to axs[2]
        cbar = fig.colorbar(cax, ax=axs[2], orientation='vertical')
        cbar.set_label('Intensity [16bit]')

        ##### Histogram

        vol_flat = vol.flatten()


        # Plot histogram with plasma colormap, where color is based on the integer value
        counts, bins, patches = plt.hist(vol_flat, bins=10, edgecolor='white')

        #  Normalize the values for the colormap
        norm = plt.Normalize(vmin=min(vol_flat), vmax=max(vol_flat))

        # Step 5: Apply the 'plasma' colormap to the histogram bars
        cmap = plt.get_cmap(default_cmap)

        for count, patch in zip(counts, patches):
            color = cmap(norm(patch.xy[0]))  # Color based on bin start (patch.xy[0] is the bin start)
            patch.set_facecolor(color)

        # Step 6: Add color bar for reference (pass mappable directly)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # No need to set array if not using an image

        # Step 7: Create colorbar using the mappable (sm) object
        plt.colorbar(sm, ax=plt.gca(), label="Integer Values")

        # Show plot
        axs[3].set_xlabel("Integer Value")
        axs[3].set_ylabel("Frequency")
        axs[3].set_title("Histogram")


        fig.tight_layout()
        fig.savefig(os.path.join(result_path,scan_name + "." + algorithm + ".vol.pdf"),dpi=300)
    

        print("#===========saving volume=========================")
        print(f"saving reconstruction file {os.path.join(path,scan_name + '.vol.h5')}")
        with h5py.File(os.path.join(path,scan_name + "." + algorithm + '.vol.h5'), 'w') as f:
            # Create a dataset in the file
            f.create_dataset('volume', data=vol,compression="gzip")



    else:
        print(f"config file not found in path {path}")
        print("A CERA config file must be provided")




if __name__ == "__main__":
    main()