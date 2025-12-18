#!/usr/bin/env python3
"""
Gaussian Density Analysis Module
================================

This module provides functionality to calculate and visualize Gaussian density profiles
for atomic systems. It is designed to analyze particle distributions and their evolution
over time, particularly useful for studying sintering, oxidation, and other dynamic processes.

Key Features:
- Calculation of 2D Gaussian density maps from atomic coordinates.
- Handling of periodic boundary conditions (PBC).
- Clustering analysis to isolate specific particles or regions.
- Visualization tools for density evolution.

Dependencies:
- numpy
- ase
- freud
- matplotlib
- tqdm
"""

import numpy as np
import tqdm
import multiprocessing as mp
from functools import partial
import pickle
import os
import freud
from ase import Atoms
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

def extract_atomic_data_from_frame(frame_data):
    """
    Extracts atomic data from a frame dictionary.

    Args:
        frame_data (dict): Dictionary containing frame information.
                           Expected keys: 'ids', 'type', 'position', 'box'.

    Returns:
        tuple: (all_ids, all_elements, all_positions, box_dims)
    """
    # Placeholder implementation - adapt to actual data structure
    # This function assumes frame_data is a dictionary with specific keys.
    # You might need to adjust this based on how your data is actually stored.
    all_ids = np.array(frame_data.get('ids', []))
    all_elements = np.array(frame_data.get('type', []))
    all_positions = np.array(frame_data.get('position', []))
    box_dims = np.array(frame_data.get('box', [100, 100, 100])) # Default box if missing
    return all_ids, all_elements, all_positions, box_dims

def calculate_gaussian_density_for_frame(frame_data, ref_element, cutoff, D, r_max, sigma, L):
    """
    Calculates Gaussian density for a single frame.

    This function identifies a cluster of atoms around a reference element,
    recenters the cluster, and computes the 2D Gaussian density.

    Args:
        frame_data (dict): Data for a single frame.
        ref_element (str): The element to build the cluster around (e.g., 'Al').
        cutoff (float): The cutoff radius to find neighboring atoms.
        D (int): The number of bins per side for the density grid (total bins = D*D).
        r_max (float): The maximum radius for the Gaussian density calculation.
        sigma (float): The standard deviation for the Gaussian smearing.
        L (float): The side length of the cubic analysis box for freud.

    Returns:
        np.ndarray: A 2D array representing the computed Gaussian density. 
                    Returns None if the frame is invalid or no cluster is found.
    """
    try:
        all_ids, all_elements, all_positions, box_dims = extract_atomic_data_from_frame(frame_data)

        # 1. Identify reference atoms and other atoms
        is_ref = (all_elements == ref_element)
        is_not_ref = ~is_ref

        ref_positions = all_positions[is_ref]
        other_positions = all_positions[is_not_ref]
        other_indices = np.where(is_not_ref)[0]

        if ref_positions.shape[0] == 0:
            return None # No reference atoms in this frame

        # 2. Find indices of 'other' atoms within the cutoff distance of *any* reference atom
        if other_positions.shape[0] > 0:
            dist_vectors = other_positions[np.newaxis, :, :] - ref_positions[:, np.newaxis, :]
            dist_vectors -= box_dims * np.round(dist_vectors / box_dims) # Apply PBC
            squared_distances = np.sum(dist_vectors**2, axis=2)
            
            # For each 'other' atom, find its minimum distance to any 'ref' atom
            min_sq_dists_to_ref = np.min(squared_distances, axis=0)
            
            # Get the original indices of 'other' atoms that are within the cutoff
            nearby_other_indices = other_indices[min_sq_dists_to_ref < cutoff**2]
        else:
            nearby_other_indices = np.array([], dtype=int)

        # 3. Fabricate the cluster
        ref_indices = np.where(is_ref)[0]
        cluster_indices = np.union1d(ref_indices, nearby_other_indices) # Use union to avoid duplicates
        
        cluster_positions = all_positions[cluster_indices]
        cluster_elements = all_elements[cluster_indices]

        if cluster_positions.shape[0] == 0:
            return None

        # 4. Recenter the cluster using ASE to correctly handle PBC
        cluster_atoms = Atoms(symbols=cluster_elements,
                              positions=cluster_positions,
                              cell=np.diag(box_dims), # Orthorhombic box
                              pbc=True)
        
        # ASE's get_center_of_mass handles atoms wrapped across periodic boundaries
        centroid = cluster_atoms.get_center_of_mass()
        # Recenter the original positions (not the wrapped ones from ASE)
        recentered_positions = cluster_positions - centroid
        
        freud_box = freud.box.Box(Lx=L, Ly=L, Lz=L, is2D=False)
        aq = freud.AABBQuery(freud_box, recentered_positions)
        
        # Compute Gaussian density
        Bins = D * D 
        gd = freud.density.GaussianDensity(Bins, r_max=r_max, sigma=sigma)
        gd.compute(system=aq)
        
        return gd.density
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

def process_trajectory(frames_data, nproc=4, ref_element='Al', cutoff=3.0, D=10, r_max=5.0, sigma=1.0, L=50.0):
    """
    Process a trajectory of frames in parallel to calculate Gaussian densities.

    Args:
        frames_data (list): List of frame data dictionaries.
        nproc (int): Number of processes to use.
        ref_element (str): Reference element.
        cutoff (float): Cutoff radius.
        D (int): Bins per side.
        r_max (float): Max radius for density.
        sigma (float): Sigma for Gaussian.
        L (float): Box size.

    Returns:
        list: List of Gaussian density arrays.
    """
    worker_function = partial(calculate_gaussian_density_for_frame,
                              ref_element=ref_element,
                              cutoff=cutoff,
                              D=D,
                              r_max=r_max,
                              sigma=sigma,
                              L=L)

    g_densities = []
    print(f"Starting Gaussian Density calculation on {len(frames_data)} frames using {nproc} processes...")
    
    with mp.Pool(processes=nproc) as pool:
        results_iterator = pool.imap(worker_function, frames_data)
        for result in tqdm.tqdm(results_iterator, total=len(frames_data)):
            if result is not None:
                g_densities.append(result)
                
    return g_densities

def plot_density_evolution(g_densities, output_path='density_evolution.png', 
                          start_time=0, time_interval=100, 
                          l_box=50.0, d_bins=10, layout=(5, 4)):
    """
    Plot the evolution of Gaussian density over time.

    Args:
        g_densities (list): List of density arrays.
        output_path (str): Path to save the figure.
        start_time (float): Simulation time of the first frame (ps).
        time_interval (float): Time interval between plotted frames (ps).
        l_box (float): Box size used in calculation (Angstrom).
        d_bins (int): Number of bins used in calculation.
        layout (tuple): (rows, cols) for the subplot grid.
    """
    if not g_densities:
        print("No density data to plot.")
        return

    rows, cols = layout
    num_plots = rows * cols
    step = max(1, len(g_densities) // num_plots)
    
    # Select frames to plot
    indices_to_plot = list(range(0, min(len(g_densities), num_plots * step), step))[:num_plots]
    
    # Calculate global vmax for consistent color scale
    # Use the middle slice of the 3D density (assuming flattened array structure related to freud)
    # Note: freud.density.GaussianDensity returns a flattened array of size D*D*D or D*D depending on setup.
    # The original code implies a 3D grid flattened, or 2D. Let's assume the input is consistent.
    # If the input is 1D (flattened), we need to reshape.
    # Freud GaussianDensity usually returns density field.
    
    # Assuming g_densities contains 3D arrays or flattened 3D arrays.
    # We need to slice them. 
    # Let's assume the user wants a cross-section.
    
    # Logic from original code: slice_mid = int(np.mean(slice_s))
    # It seems the original code was handling a specific flattened structure.
    # For general purpose, we'll assume the input is 3D (D, D, D) or we reshape it.
    
    sample_density = g_densities[0]
    total_bins = sample_density.size
    # Check if it's a cube
    d_calculated = int(round(total_bins**(1/3)))
    
    if d_calculated**3 == total_bins:
         # Reshape to 3D
         reshaped_densities = [d.reshape((d_calculated, d_calculated, d_calculated)) for d in g_densities]
         mid_slice = d_calculated // 2
         slices = [d[mid_slice, :, :] for d in reshaped_densities]
    else:
        # Assume 2D or already sliced?
        # If it's D*D
        d_2d = int(round(total_bins**(1/2)))
        if d_2d**2 == total_bins:
             slices = g_densities
        else:
             print("Warning: Density array shape not recognized as cube or square. Plotting might be incorrect.")
             slices = g_densities

    # Calculate Vmax
    max_vals = [s.max() for s in slices]
    vmax = np.mean(max_vals)

    # Custom Colormap
    colors = [(1.0, 1.0, 1.0, 0), 
              '#eaea94', '#e6db65', 
              '#e6cb3a', '#e6b917', 
              '#e69e0c', '#e68301', 
              '#ef680e', '#fa4d1e', 
              '#ea3d39', '#bf385f', 
              '#953283', '#752d9e', 
              '#5528b9', '#4026ab', 
              '#302690', '#000000']
    cmap_name = "custom_thermal"
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), dpi=120)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(indices_to_plot):
            idx = indices_to_plot[i]
            density_slice = slices[idx]
            
            im = ax.imshow(density_slice, cmap=custom_cmap, vmin=0, vmax=vmax, origin='lower', extent=[0, l_box, 0, l_box])
            
            current_time = start_time + idx * time_interval # Approximate time
            ax.set_title(f'$t$: {current_time:.1f} ps', fontsize=14)
            ax.set_xlabel('$L$ (Å)')
            ax.set_ylabel('$L$ (Å)')
        else:
            ax.axis('off')

    # Colorbar
    cax = fig.add_axes([0.2, 0.96, 0.6, 0.02])
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('Gaussian Density (a.u.)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200, transparent=True)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    print("This module is intended to be imported or run with specific data.")
    print("Example usage:")
    print("  from gaussian_density import process_trajectory, plot_density_evolution")
    print("  densities = process_trajectory(frames)")
    print("  plot_density_evolution(densities)")
