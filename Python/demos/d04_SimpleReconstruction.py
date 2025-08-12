#%% Demo 4: Simple Image reconstruction
#
#
# This demo will show how a simple image reconstruction can be performed,
# by using OS-SART and FDK
#
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# This file is part of the TIGRE Toolbox
#
# Copyright (c) 2015, University of Bath and
#                     CERN-European Organization for Nuclear Research
#                     All rights reserved.
#
# License:            Open Source under BSD.
#                     See the full license at
#                     https://github.com/CERN/TIGRE/blob/master/LICENSE
#
# Contact:            tigre.toolbox@gmail.com
# Codes:              https://github.com/CERN/TIGRE/
# Coded by:           Ander Biguri
# --------------------------------------------------------------------------
#%%Initialize
import tigre
import numpy as np
from tigre.utilities import sample_loader
from tigre.utilities import CTnoise
import tigre.algorithms as algs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#%% Geometry
geo = tigre.geometry_default(high_resolution=False)

#%% Load data and generate projections
# define angles
angles = np.linspace(0, 2 * np.pi, 100)
# Load thorax phatom data
head = sample_loader.load_head_phantom(geo.nVoxel)
# generate projections
projections = tigre.Ax(head, geo, angles)
# add noise
noise_projections = CTnoise.add(projections, Poisson=1e5, Gaussian=np.array([0, 10]))

print(noise_projections.shape)
print(noise_projections.dtype)
print(noise_projections.min(),noise_projections.max())

fig , axs = plt.subplots(1,2,figsize=(2*4.8,4.8))
axs = axs.flatten()
cax = axs[0].imshow(noise_projections[:, noise_projections.shape[1]//2, :], cmap="plasma_r", aspect='auto',norm=LogNorm())
axs[0].set_xlabel(r"$u$ [Px]")
axs[0].set_ylabel(r"$\theta$ [rad]")

# Add a colorbar to axs[0]
cbar = fig.colorbar(cax, ax=axs[0], orientation='vertical')
cbar.set_label(r'$\log{(I)}$ [16bit]')
# Replace y-axis with theta values
num_ticks = 10
tick_positions = np.linspace(0, noise_projections.shape[0] - 1, num_ticks).astype(int)
tick_labels = [f"{angles[i]:.2f}" for i in tick_positions]
axs[0].set_yticks(tick_positions)
axs[0].set_yticklabels(tick_labels)




fig.tight_layout()
fig.savefig("test.sino.pdf",dpi=300)


#%% Reconstruct image using OS-SART and FDK

# FDK
imgFDK = algs.fdk(noise_projections, geo, angles)
# OS-SART

# niter = 50
# imgOSSART = algs.ossart(noise_projections, geo, angles, niter)

#%% Show the results
# tigre.plotimg(np.concatenate([imgFDK, imgOSSART], axis=1), dim="z")
