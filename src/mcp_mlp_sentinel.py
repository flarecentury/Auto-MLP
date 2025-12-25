import asyncio
import os
import sys
import time
import datetime
import pickle
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import multiprocessing as mp
from deepmd.calculator import DP
from dpdata import LabeledSystem, MultiSystems
from fastmcp import FastMCP
from ase import Atoms

# --- Environment Setup (from source.py) ---
# Note: we are keeping the strict paths as requested
conda_env = 'deepmd3.0.3'
os.environ['PATH'] = f"/storage/MD_domain/miniconda3/envs/{conda_env}/bin{os.pathsep}{os.environ['PATH']}"
os.environ['LAMMPS_PLUGIN_PATH'] = f'/storage/MD_domain/miniconda3/envs/{conda_env}/lib/deepmd_lmp'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# --- Constants & Config ---
MODEL_PATH = 'models/frozen_model.pth' # Implied from ref_mcp.py and user request
# Use the path from source.py for data
pkldir = 'pkl'
DATA_PATH_TEMPLATE = '/storage/MD_domain/Deep_MD/AIMD/DPGEN_sampling/2_DP_RUN_DSH_soap/finetune/splitdata/{}_data'
FILTER_LIST = ['Al0O2', 'Al0O3', 'Al0O4', 'Al0O6', 'Al0O8', 'Al0O54']

colors_dict = {
    'black':   '#343434',
    'orange':  '#f7c242',
    'green':   '#93c572',
    'pink':    '#f49ac2',
    'gray':    '#b8b8b8',
    'purple':  '#6b5b95',
    'red':     '#d64161',
}
black = colors_dict['black']
orange = colors_dict['orange']
green = colors_dict['green']
pink = colors_dict['pink']
gray = colors_dict['gray']
purple = colors_dict['purple']
red = colors_dict['red']


# --- Helper Functions (Adapted from source.py) ---

def process_structures_on_gpu(args):
    """Worker function for multiprocessing."""
    gpu_id, structure_batch, model_path, head = args # Pass model path instead of model obj
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Load model inside process
    try:
         # DP might fail if model path is invalid, handle gracefully if possible
        call = DP(model=model_path)
    except Exception as e:
        # Check if we can just use the provided path, usually strict
        call = DP(model=model_path)

    indexed_origin_results = []
    indexed_results = []

    for original_idx, atoms in structure_batch:
        # Robustly ensure 'atoms' is an ASE Atoms object
        if not isinstance(atoms, Atoms):
            # If it is a dpdata System/LabeledSystem, convert it.
            if hasattr(atoms, 'to_ase_structure'):
                try:
                    res = atoms.to_ase_structure()
                    if isinstance(res, list):
                        atoms = res[0]
                    else:
                        atoms = res
                except Exception:
                    pass
        
        # Final safety check
        if not hasattr(atoms, 'get_potential_energy'):
             continue

        try:
            origin_s = LabeledSystem().from_ase_structure(atoms, fmt='ase/structure')
        except:
             # If conversion fails, just calc
            pass
            
        # Calculation
        atoms.calc = call
        atoms.get_potential_energy()
        atoms.get_forces()
        
        predicted_s = LabeledSystem().from_ase_structure(atoms, fmt='ase/structure')
        indexed_results.append((original_idx, predicted_s))
        
        # re-create origin to be sure (source.py logic)
        origin_s = LabeledSystem().from_ase_structure(atoms, fmt='ase/structure')
        indexed_origin_results.append((original_idx, origin_s))

    return (indexed_origin_results, indexed_results)


def process_structures_by_dp(sys_data, model_path, gpu_ids=None, batchnum=500):
    if gpu_ids is None:
        gpu_ids = [0]
    
    atom_list = []
    if isinstance(sys_data, MultiSystems):
        for s in sys_data:
            if hasattr(s, 'to_ase_structure'):
                res = s.to_ase_structure()
                if isinstance(res, list):
                    atom_list.extend(res)
                else:
                    atom_list.append(res)
            else:
                atom_list.append(s)
    elif hasattr(sys_data, 'to_ase_structure'):
        try:
            res = sys_data.to_ase_structure()
            if isinstance(res, list):
                atom_list = res
            else:
                atom_list = [res]
        except Exception:
            if hasattr(sys_data, '__iter__') and not isinstance(sys_data, (str, bytes)):
                 for item in sys_data:
                     if hasattr(item, 'to_ase_structure'):
                         res = item.to_ase_structure()
                         if isinstance(res, list):
                             atom_list.extend(res)
                         else:
                             atom_list.append(res)
                     else:
                         atom_list.append(item)
            else:
                 atom_list = [sys_data]
    elif isinstance(sys_data, (list, tuple)):
        for item in sys_data:
            if hasattr(item, 'to_ase_structure'):
                res = item.to_ase_structure()
                if isinstance(res, list):
                    atom_list.extend(res)
                else:
                    atom_list.append(res)
            else:
                atom_list.append(item)
    else:
        try:
            atom_list = [s for s in sys_data]
        except:
            atom_list = [sys_data]

    # Setup batches
    gpu_batches = [[] for _ in range(len(gpu_ids))]
    if len(atom_list) < batchnum:
        for i, atoms in enumerate(atom_list):
            gpu_batches[0].append((i, atoms))
        tasks = [(gpu_ids[0], gpu_batches[0], model_path, '')]
        n_processes = 1
    else:
        for i, atoms in enumerate(atom_list):
            gpu_idx_local = i % len(gpu_ids)
            gpu_batches[gpu_idx_local].append((i, atoms))
        tasks = [(gpu_id, gpu_batches[j], model_path, '') for j, gpu_id in enumerate(gpu_ids) if gpu_batches[j]]
        n_processes = len(tasks)

    # Run pool
    # Note: source.py uses 'fork' or 'spawn'. We'll stick to 'spawn' for CUDA safety or default.
    # source.py had: mp.set_start_method('fork', force=True)
    # We will trust the environment or set it if needed. 
    # But inside a tool, setting start method might be risky if already set.
    # We'll try to use the context.
    ctx = mp.get_context("fork") # source.py explicitly used fork
    
    all_indexed_origin_results_tuples = []
    all_indexed_final_results_tuples = []

    with ctx.Pool(processes=n_processes) as pool:
        for batch_origin_tuples, batch_final_tuples in pool.imap(process_structures_on_gpu, tasks):
            all_indexed_origin_results_tuples.extend(batch_origin_tuples)
            all_indexed_final_results_tuples.extend(batch_final_tuples)

    # Sort
    all_indexed_origin_results_tuples.sort(key=lambda x: x[0])
    all_indexed_final_results_tuples.sort(key=lambda x: x[0])
    
    sorted_origin_frames = [ls_frame for _, ls_frame in all_indexed_origin_results_tuples]
    sorted_final_frames = [ls_frame for _, ls_frame in all_indexed_final_results_tuples]

    origin_sys_ordered = MultiSystems()
    if sorted_origin_frames:
        for f in sorted_origin_frames:
            origin_sys_ordered.append(f)
        
    final_sys_ordered = MultiSystems()
    if sorted_final_frames:
        for f in sorted_final_frames:
            final_sys_ordered.append(f)
        
    return origin_sys_ordered, final_sys_ordered

def ms_filter(ms, filter_list):
    newms = MultiSystems()
    for sys in ms:
        if sys.formula not in filter_list:
            newms.append(sys)
    return newms

# Error metrics
mae = lambda x: np.mean(np.abs(x))
rmse = lambda x: np.sqrt(np.mean(x * x))
def r2_score(y_true, y_pred):
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0

def calculate_metrics(training_systems_list, predict_list):
    results = {
        "training_energies": [], "predicted_energies": [],
        "training_energies_a": [], "predicted_energies_a": [],
        "training_forces": [], "predicted_forces": [],
        "atom_numbers": []
    }
    total_atoms = 0
    total_frames = 0
    
    for training_systems, predict in zip(training_systems_list, predict_list):
        atom_num = sum(training_systems.data['atom_numbs'])
        
        true_e = training_systems["energies"]
        pred_e = predict["energies"]
        true_f = training_systems["forces"]
        pred_f = predict["forces"]
        
        true_e_a = true_e / atom_num
        pred_e_a = pred_e / atom_num
        
        results["training_energies"].extend(true_e)
        results["predicted_energies"].extend(pred_e)
        results["training_energies_a"].extend(true_e_a)
        results["predicted_energies_a"].extend(pred_e_a)
        results["training_forces"].append(true_f)
        results["predicted_forces"].append(pred_f)
        results["atom_numbers"].append(atom_num)
        
        total_atoms += atom_num * len(true_e)
        total_frames += len(true_e)

    results["training_energies"] = np.array(results["training_energies"])
    results["predicted_energies"] = np.array(results["predicted_energies"])
    results["training_energies_a"] = np.array(results["training_energies_a"])
    results["predicted_energies_a"] = np.array(results["predicted_energies_a"])
    
    tf_flat = [f.reshape(-1, 3) for f in results["training_forces"]]
    pf_flat = [f.reshape(-1, 3) for f in results["predicted_forces"]]
    results["training_forces"] = np.vstack(tf_flat)
    results["predicted_forces"] = np.vstack(pf_flat)
    
    results["energy_diff"] = results["predicted_energies"] - results["training_energies"]
    results["force_diff"] = results["predicted_forces"] - results["training_forces"]
    
    results["mae_ea"] = mae(results["energy_diff"]) / (total_atoms / total_frames)
    results["rmse_ea"] = rmse(results["energy_diff"]) / (total_atoms / total_frames)
    results["mae_f"] = mae(results["force_diff"])
    results["rmse_f"] = rmse(results["force_diff"])
    
    return results

# --- Plotting Functions ---

def fig_to_image(fig):
    """Convert matplotlib figure to MCP Image (dict)"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, transparent=True)
    buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return {
        "type": "image",
        "data": data,
        "mimeType": "image/png"
    }

def generate_plots(results):
    images = []
    
    # 1. Energy: Predicted vs True
    fig_energy, ax_energy = plt.subplots(figsize=(8, 6), dpi=150)
    r2_e = r2_score(results["training_energies_a"], results["predicted_energies_a"])
    
    ax_energy.scatter(results["training_energies_a"], results["predicted_energies_a"], c=purple, alpha=0.5)
    lim = [min(results["training_energies_a"].min(), results["predicted_energies_a"].min()),
           max(results["training_energies_a"].max(), results["predicted_energies_a"].max())]
    ax_energy.plot(lim, lim, 'k--', label='Perfect prediction')
    ax_energy.set_xlabel('$E_{\\text{DFT}}$ (eV/atom)')
    ax_energy.set_ylabel('$E_{\\text{MLP}}$ (eV/atom)')
    ax_energy.set_title('Energy: Predicted vs True')
    stats_text = f'MAE: {results["mae_ea"]:.3f} eV/atom\nRMSE: {results["rmse_ea"]:.3f} eV/atom\nR²: {r2_e:.3f}'
    ax_energy.text(0.05, 0.95, stats_text, transform=ax_energy.transAxes, fontsize=12, va='top')
    ax_energy.legend()
    ax_energy.grid(True, linestyle='--', alpha=0.7)
    
    # Inset
    axins_energy = inset_axes(ax_energy, width="30%", height="30%", loc='upper left',
                              bbox_to_anchor=(0.65, -0.44, 1, 1), bbox_transform=ax_energy.transAxes)
    data = results["energy_diff"].flatten()
    filtered_data = data[data <= np.percentile(data, 98)]
    counts, bin_edges = np.histogram(filtered_data, bins=30)
    percentages = (counts / len(filtered_data)) * 100
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    axins_energy.bar(bin_centers, percentages, width=np.diff(bin_edges), color=purple, alpha=0.7)
    axins_energy.set_title('Distribution of Error', fontsize=15)
    axins_energy.set_ylabel('%', fontsize=15)
    axins_energy.set_xlabel('(eV/atom)', fontsize=15)
    
    images.append(fig_to_image(fig_energy))
    plt.close(fig_energy)

    # 2. Force: Predicted vs True
    fig_force, ax_force = plt.subplots(figsize=(8, 6), dpi=150)
    r2_f = r2_score(results["training_forces"].flatten(), results["predicted_forces"].flatten())
    
    # Sample if too large
    tf = results["training_forces"].flatten()
    pf = results["predicted_forces"].flatten()
    indices = np.arange(len(tf))
    if len(tf) > 10000:
        indices = indices[::int(len(tf)/10000)]
    
    ax_force.scatter(tf[indices], pf[indices], c=green, alpha=0.5)
    lim = [min(tf.min(), pf.min()), max(tf.max(), pf.max())]
    ax_force.plot(lim, lim, 'k--', label='Perfect prediction')
    ax_force.set_xlabel('$f_{\\text{DFT}}$ (eV/Å)')
    ax_force.set_ylabel('$f_{\\text{MLP}}$ (eV/Å)')
    ax_force.set_title('Force: Predicted vs True')
    stats_text = f'MAE: {results["mae_f"]:.3f} eV/Å\nRMSE: {results["rmse_f"]:.3f} eV/Å\nR²: {r2_f:.3f}'
    ax_force.text(0.05, 0.95, stats_text, transform=ax_force.transAxes, fontsize=12, va='top')
    ax_force.legend()
    ax_force.grid(True, linestyle='--', alpha=0.7)
    
    # Inset
    axins_force = inset_axes(ax_force, width="30%", height="30%", loc='upper left',
                             bbox_to_anchor=(0.65, -0.44, 1, 1), bbox_transform=ax_force.transAxes)
    data = results["force_diff"].flatten()
    filtered_data = data[data <= np.percentile(data, 98)]
    counts, bin_edges = np.histogram(filtered_data, bins=30)
    percentages = (counts / len(filtered_data)) * 100
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    axins_force.bar(bin_centers, percentages, width=np.diff(bin_edges), color=green, alpha=0.7)
    axins_force.set_title('Distribution of Error', fontsize=15)
    axins_force.set_ylabel('%', fontsize=15)
    axins_force.set_xlabel('(eV/Å)', fontsize=15)
    
    images.append(fig_to_image(fig_force))
    plt.close(fig_force)

    # 3. Force Components
    fig_comp = plt.figure(figsize=(8, 6), dpi=150)
    stats_text = "RMSE\n"
    r2_text = "R²\n"
    for i, (color, comp) in enumerate([(red, 'x'), (orange, 'y'), (purple, 'z')]):
        true_comp = results["training_forces"][:, i].flatten()
        pred_comp = results["predicted_forces"][:, i].flatten()
        
        # Sample
        indices = np.arange(len(true_comp))
        if len(indices) > 10000:
             indices = indices[::int(len(indices)/10000)]
             
        plt.scatter(true_comp[indices], pred_comp[indices], c=color, alpha=0.5, label=f'f{comp}')
        
        rmse_val = rmse(pred_comp - true_comp)
        r2_val = r2_score(true_comp, pred_comp)
        stats_text += f'$f_{comp}$: {rmse_val:.3f} eV/Å\n'
        r2_text += f'$f_{comp}$: {r2_val:.3f}\n'

    lim = [results["training_forces"].min(), results["training_forces"].max()]
    plt.plot(lim, lim, 'k--', label='Perfect prediction')
    plt.xlabel('$F_{\\text{DFT}}$ (eV/Å)')
    plt.ylabel('$F_{\\text{MLP}}$ (eV/Å)')
    plt.title('Force Components: Predicted vs True')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.text(0.05, 0.95, f"{stats_text}\n{r2_text}", transform=plt.gca().transAxes, fontsize=15, va='top')
    
    images.append(fig_to_image(fig_comp))
    plt.close(fig_comp)
    
    return images

# --- MCP Server ---

mcp = FastMCP("Auto Test MCP")

@mcp.tool()
def run_auto_test(test_mode: bool = True) -> list:
    """
    Run automated tests on the current version of the force field model. model haven't been updated for a while, so you should always use the test_mode=True
    Args:
        test_mode: If True, load prediction results from 'ms_test_dp.pkl' instead of running inference.
    Returns generated performance charts, test duration, and model timestamp.
    """
    start_time = time.time()
    
    print(f"Starting Auto Test Tool (test_mode={test_mode})...")
    
    # 1. Check model timestamp
    if not os.path.exists(MODEL_PATH):
        return [f"Error: Model file not found at {MODEL_PATH}"]
    
    model_mtime = os.path.getmtime(MODEL_PATH)
    model_timestamp_str = datetime.datetime.fromtimestamp(model_mtime).strftime('%Y-%m-%d %H:%M:%S')

    print(f"Model found. Timestamp: {model_timestamp_str}")
    print("Loading Test Data...")

    # 2. Load Data (Test Set)
    dataset_label = 'test'
    data_path = DATA_PATH_TEMPLATE.format(dataset_label)
    
    # Try loading from deepmd format first (source.py logic)
    try:
        if os.path.exists(data_path) or os.path.exists(data_path + '.npy'): # basic check
             ms_dft = MultiSystems().load_systems_from_file(file_name=data_path, fmt='deepmd/npy')
        else:
            # Fallback to pickle if user cached it as in source.py
             ms_dft = pickle.load(open(f'{pkldir}/ms_{dataset_label}_dft.pkl', 'rb'))
    except Exception as e:
        return [f"Error loading data: {e}"]

    # Filter
    ms_dft = ms_filter(ms_dft, filter_list=FILTER_LIST)
    print(f"Loaded {len(ms_dft)} systems for testing.")

    # 3. Run Inference
    if test_mode:
        print("Test Mode: Loading cached predictions from ms_test_dp.pkl...")
        try:
            # Assuming file name ms_test_dp.pkl as requested
            ms_predict = pickle.load(open(f'{pkldir}/ms_{dataset_label}_predict_dp.pkl', 'rb'))
        except Exception as e:
            return [f"Error loading cached predictions: {e}"]
    else:
        print("Running Inference on GPU...")
        try:
            _, ms_predict = process_structures_by_dp(ms_dft, MODEL_PATH, gpu_ids=[0, 1])
        except Exception as e:
            return [f"Error during inference: {e}"]
        
    ms_predict = ms_filter(ms_predict, filter_list=FILTER_LIST)

    # 4. Calculate Metrics
    print("Calculating metrics...")
    results = calculate_metrics(ms_dft, ms_predict)
    
    # 5. Generate Plots
    print("Generating plots...")
    plot_images = generate_plots(results)
    
    end_time = time.time()
    duration = end_time - start_time
    
    if test_mode:
        summary_text = f"""### Test Complete (model not changed, display the results from last test)
- **Model Timestamp**: {model_timestamp_str}
- **Test Duration**: {duration:.2f} seconds
- **Energy MAE**: {results['mae_ea']:.4f} eV/atom
- **Force MAE**: {results['mae_f']:.4f} eV/Å
    """
    else:
        summary_text = f"""### Test Complete
- **Model Timestamp**: {model_timestamp_str}
- **Test Duration**: {duration:.2f} seconds
- **Energy MAE**: {results['mae_ea']:.4f} eV/atom
- **Force MAE**: {results['mae_f']:.4f} eV/Å
    """
    # FastMCP should generally handle a list of mixed content type (Text + Image/Resource)
    # The image dicts are standard MCP image content objects.
    return [summary_text] + plot_images

def initialize_model():
    """Check model availability at startup."""
    if os.path.exists(MODEL_PATH):
        print(f"Model found at {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

def main():
    """Main function to run the MCP server with transport options."""
    # Initialize model at startup
    initialize_model()
    
    # Parse command line arguments for transport type
    transport = "http"  # Default transport as requested by user ("default http streamble mcp")
    host = "0.0.0.0"
    port = 8000
    
    # Simple argument parsing to match user reference
    if len(sys.argv) > 1:
        if sys.argv[1] == "--http":
            transport = "http"
        elif sys.argv[1] == "--sse":
            transport = "sse"
        elif sys.argv[1] == "--stdio":
            transport = "stdio"
    
    # Parse additional arguments for host and port
    for i, arg in enumerate(sys.argv):
        if arg == "--host" and i + 1 < len(sys.argv):
            host = sys.argv[i + 1]
        elif arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
    
    print(f"Starting MCP server with transport: {transport}")
    
    # Ensure multiprocessing safety
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass

    if transport == "stdio":
        # Run with stdio transport
        mcp.run()
    elif transport == "http":
        # Run with HTTP streamable transport
        print(f"HTTP server starting at http://{host}:{port}/mcp/")
        mcp.run(transport="http", host=host, port=port)
    elif transport == "sse":
        # Run with SSE transport
        print(f"SSE server starting at http://{host}:{port}/sse")
        mcp.run(transport="sse", host=host, port=port)

if __name__ == "__main__":
    main()
