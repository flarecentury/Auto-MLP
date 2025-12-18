# Auto-MLP: AI-Driven Machine Learning Potential Development Framework

## Quick Start: Online AI Agent Demo

Experience the AI-driven research assistant immediately without installation:
**[Try the AI Agent Online](https://research-helper.digauto.org)**

---

## Overview

**Auto-MLP** is a comprehensive framework designed to automate the development of Machine Learning Potentials (MLPs) for complex atomic systems. It integrates active learning strategies, high-throughput density functional theory (DFT) calculations, and deep learning model training into a closed-loop system.

This repository contains the core computational modules and analysis scripts used in our research on **AI-driven discovery of combustion mechanisms in aluminum nanoparticles**.

**Key Capabilities:**
*   **Active Learning:** Automated exploration of chemical space using uncertainty quantification (Query-by-Committee).
*   **DeepMD Integration:** Seamless interface with DeepMD-kit for training and inference.
*   **Advanced Analysis:** Tools for Gaussian density analysis, SOAP descriptor calculation, and structural visualization.
*   **MCP Support:** Implements the Model Context Protocol (MCP) for integration with AI agents.

## Repository Structure

```
Auto-MLP/
├── src/                # Core source code
│   ├── gaussian_density.py  # Gaussian density analysis for particle evolution
│   ├── soap_analysis.py     # SOAP descriptor calculation & diversity selection
│   ├── mcp_viz.py           # MCP server for 3D atomic visualization
│   └── mcp_inference.py     # MCP server for MLP inference (DeepMD)
├── data/               # Sample data (structures, models)
├── docs/               # Documentation
├── examples/           # Usage examples
├── scripts/            # Utility scripts
└── README.md           # This file
```

## Installation

### Prerequisites

*   Python 3.8+
*   PyTorch (with CUDA support for GPU acceleration)
*   DeepMD-kit (for MLP training/inference)
*   ASE (Atomic Simulation Environment)

### Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/flarecentury/Auto-MLP.git
    cd Auto-MLP
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Gaussian Density Analysis
Analyze the evolution of particle morphology during sintering or oxidation.

```python
from src.gaussian_density import process_trajectory, plot_density_evolution

# Load your trajectory (list of frame dicts)
frames = load_trajectory("simulation.dump") 
densities = process_trajectory(frames, nproc=4)
plot_density_evolution(densities, output_path="density.png")
```

### 2. Active Learning with SOAP
Select diverse structures from a large MD trajectory for DFT labeling.

```python
from src.soap_analysis import select_diverse_structures

# Select structures with high diversity
selected = select_diverse_structures(all_structures, descriptor_calc, s_max=0.05)
```

### 3. MCP Servers
Start the MCP servers to enable AI agent interaction.

```bash
# Visualization Server
python src/mcp_viz.py

# Inference Server
python src/mcp_inference.py
```

## Data Availability

The AI agent is now available online at [https://research-helper.digauto.org](https://research-helper.digauto.org).

Key computational code including the MCP module for our closed-loop AI agent framework, trained machine learning potential (MLP) models, and analysis scripts are available in this GitHub repository.

The comprehensive MLP training dataset containing approximately 90,000 atomic configurations with corresponding DFT energies and forces is hosted on the **Digital Automation for Scientific Discovery** platform (DigAuto): [https://www.digauto.org](https://www.digauto.org).

## Citation

If you use this code in your research, please cite our work:

> Self-Reinforcing AI Agents Accelerate Active Learning ML Potential-Driven Aluminum Nanoparticle Combustion Simulations
> 
> **Authors:**
> - **Yiming Lu**
> - **Tingyu Lu**
> - **Zhang Di**
> - **Lili Ye** (Corresponding Author)
> - **Hao Li** (Corresponding Author)
> - **Mingshu Bi**


## License

This project is licensed under the MIT License.

## Contact

For questions or collaboration, please contact the corresponding authors via the [DigAuto Platform](https://www.digauto.org).
