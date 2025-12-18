#!/usr/bin/env python3
"""
SOAP Descriptor Analysis Module
===============================

This module provides tools for calculating Smooth Overlap of Atomic Positions (SOAP)
descriptors and performing similarity analysis on atomic structures. It includes
optimized GPU-accelerated similarity calculations.

Key Features:
- SOAP descriptor calculation using DScribe (or similar libraries).
- GPU-accelerated cosine similarity calculation.
- Structure selection based on diversity (MaxMin algorithm).
- Data management for large datasets.

Dependencies:
- numpy
- torch
- ase
- tqdm
- dscribe (implied for descriptor generation)
"""

import numpy as np
import torch
import time
import tqdm
import pickle
import os
from ase.io import read

class GPUDescriptorCache:
    """
    Manages descriptor data on GPU(s) to optimize memory transfer and access.
    """
    def __init__(self):
        self.cached_descriptors = []
        self.current_gpu = 0 
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus == 0:
            print("Warning: No GPU detected. GPU acceleration will not be available.")
    
    def extend(self, new_descriptors):
        """
        Adds new descriptors to the cache, distributing them across available GPUs.
        """
        if self.num_gpus == 0:
            return

        new_descriptors = torch.from_numpy(new_descriptors)
        
        # Move new descriptors to the current GPU
        device = f'cuda:{self.current_gpu}'
        new_descriptors_gpu = new_descriptors.to(device=device)
        
        if len(self.cached_descriptors) < self.num_gpus:
            self.cached_descriptors.append(new_descriptors_gpu)
        else:
            # Concatenate with existing data on this GPU
            self.cached_descriptors[self.current_gpu] = torch.cat(
                [self.cached_descriptors[self.current_gpu], new_descriptors_gpu], dim=0)
        
        # Round-robin distribution
        self.current_gpu = (self.current_gpu + 1) % self.num_gpus
    
    def get_descriptors(self):
        return self.cached_descriptors
    
    def clear(self):
        for desc in self.cached_descriptors:
            del desc
        self.cached_descriptors = []
        if self.num_gpus > 0:
            torch.cuda.empty_cache()

def calculate_similarity_vectorized(A, gpu_cache):
    """
    Calculates the similarity (distance) between a query descriptor A and 
    a set of reference descriptors B stored in the GPU cache.

    Metric: Euclidean distance based on dot product (SOAP kernel).
    d(A, B) = sqrt( K(A,A) + K(B,B) - 2*K(A,B) )
    
    Args:
        A (np.ndarray): Query descriptor(s).
        gpu_cache (GPUDescriptorCache): Cache containing reference descriptors.

    Returns:
        tuple: (min_distances, execution_time)
    """
    start_time = time.time()
    
    B_concat_list = gpu_cache.get_descriptors()
    num_available_gpus = len(B_concat_list)
    
    if num_available_gpus == 0:
        # Fallback for CPU or empty cache
        return None, 0.0

    streams = [torch.cuda.Stream(device=i) for i in range(num_available_gpus)]
    
    A_gpus = []
    A_norms_gpus = []

    # 1. Copy Query A to all GPUs
    for i in range(num_available_gpus):
        device = f'cuda:{i}'
        with torch.cuda.device(device):
            with torch.cuda.stream(streams[i]):
                A_gpu = torch.tensor(A, device=device)
                A_norms = torch.sum(A_gpu ** 2, dim=1)
                A_gpus.append(A_gpu)
                A_norms_gpus.append(A_norms)

    all_distances_list = [None] * num_available_gpus

    # 2. Compute Distances on each GPU
    for i in range(num_available_gpus):
        device = f'cuda:{i}'
        B_concat = B_concat_list[i]
        if B_concat.shape[0] == 0:
            continue

        stream = streams[i]
        with torch.cuda.device(device):
            with torch.cuda.stream(stream):
                A_gpu = A_gpus[i]
                A_norms = A_norms_gpus[i]

                B_norms = torch.sum(B_concat ** 2, dim=2) 
                # AB = torch.matmul(B_concat, A_gpu.T)
                
                AB = torch.matmul(B_concat, A_gpu.T)

                A_norms_expanded = A_norms.unsqueeze(0).unsqueeze(0)
                B_norms_expanded = B_norms.unsqueeze(2)
                
                distances_squared = B_norms_expanded + A_norms_expanded - 2 * AB
                distances_squared = torch.clamp(distances_squared, min=0)
                distances = torch.sqrt(distances_squared)
                
                # Min distance across some dimension (e.g., best match among atoms)
                all_distances, _ = torch.min(distances, dim=2)

                all_distances_list[i] = all_distances.cpu()

    # 3. Synchronize
    for i in range(num_available_gpus):
        with torch.cuda.device(f'cuda:{i}'):
            streams[i].synchronize()

    # 4. Aggregate results
    valid_distances = [d for d in all_distances_list if d is not None]
    
    if not valid_distances:
        return None, 0.0

    all_distances = torch.cat(valid_distances, dim=0)
    
    end_time = time.time()
    return all_distances.numpy(), end_time - start_time

def split_structures(structures, chunk_size, shuffle=False):
    """
    Splits a list of structures into chunks.
    """
    if shuffle:
        import random
        random.shuffle(structures)
        
    return [structures[i:i + chunk_size] for i in range(0, len(structures), chunk_size)]

def select_diverse_structures(structures, descriptor_calc, s_max=0.05, s_ave=0.05, chunk_size=2000):
    """
    Selects a diverse set of structures using an active learning strategy (MaxMin / Uncertainty).
    
    Args:
        structures (list): List of ASE Atoms objects.
        descriptor_calc (callable): Function or object to calculate descriptors.
                                    Expected to return a dict with 'data' key.
        s_max (float): Max similarity threshold.
        s_ave (float): Average similarity threshold.
        chunk_size (int): Processing chunk size.
        
    Returns:
        list: Selected diverse structures.
    """
    chunks = split_structures(structures, chunk_size)
    selected_structures = []
    
    print(f"Starting selection with S_max={s_max}, S_ave={s_ave}")
    
    for chunk_idx, chunk in enumerate(chunks):
        if not chunk: continue
        
        # Initialize pool with the first structure
        diverse_pool = [chunk[0]]
        
        # Calculate descriptor (dummy wrapper expected)
        desc_0 = descriptor_calc.calc(chunk[0])["data"]
        diverse_pool_descriptors = np.array([desc_0])
        
        # Setup GPU Cache
        cache = GPUDescriptorCache()
        cache.clear()
        cache.extend(diverse_pool_descriptors)
        
        for i, structure in enumerate(tqdm.tqdm(chunk[1:], desc=f"Chunk {chunk_idx}", leave=False)):
            current_desc = descriptor_calc.calc(structure)["data"]
            
            # Calculate similarity
            similarities, _ = calculate_similarity_vectorized(current_desc, cache)
            
            if similarities is None:
                continue
                
            D_max = similarities.max(axis=1)
            D_ave = similarities.mean(axis=1)
            
            # Selection Criteria: If the new structure is sufficiently different from ALL existing ones
            if np.all((D_ave > s_ave) | (D_max > s_max)):
                diverse_pool.append(structure)
                cache.extend(np.array([current_desc]))
        
        selected_structures.extend(diverse_pool)
        print(f"Chunk {chunk_idx}: Selected {len(diverse_pool)} / {len(chunk)}")
        
        cache.clear()
        
    return selected_structures

if __name__ == "__main__":
    print("SOAP Analysis Module")
    print("Use this module to calculate similarities and select diverse structures.")
