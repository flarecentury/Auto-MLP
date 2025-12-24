# Auto-MLP: AI-Driven Machine Learning Potential Development Framework

## Quick Start: Online AI Agent Demo

Experience the AI-driven research assistant immediately:
**[Try the AI Agent Online](https://www.digauto.org)**

---

## Demo Video

<video src="data/videos/Video1_mlp_ai_agents.mp4" controls="controls" style="max-width: 100%;">
</video>

[Download Video](data/videos/Video1_mlp_ai_agents.mp4)

---

## Overview

This repository hosts the core computational modules and AI integration tools developed for our research: **"AI-driven discovery of combustion mechanisms in aluminum nanoparticles."**

While our ultimate vision is a fully autonomous "Auto-MLP" system for new material discovery, the current release focuses on the **AI-assisted development framework** that we have successfully implemented.

**Our Current Focus:**
*   **AI-Guided Active Learning:** Leveraging AI agents to supervise the iterative training of DeepMD potentials, significantly reducing human intervention.
*   **Combustion Mechanism Analysis:** Specialized tools for analyzing the oxidation and sintering behaviors of aluminum nanoparticles.
*   **MCP Integration:** Bridging the gap between atomic simulations and AI reasoning capabilities.

**Future Vision:**
We are actively working towards a completely closed-loop "Auto-MLP" system capable of autonomous training, validation, and discovery of novel materials without human-in-the-loop.

**Key Capabilities:**
*   **Active Learning:** Automated exploration of chemical space using uncertainty quantification (Query-by-Committee).
*   **DeepMD Integration:** Seamless interface with DeepMD-kit inference.
*   **Analysis:** Tools for Gaussian density analysis, SOAP descriptor calculation, and structural visualization.
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

## Development

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

The AI agent is now available online at [https://www.digauto.org](https://www.digauto.org).

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
> 

## License

This project is licensed under the MIT License.

## Contact

For questions or collaboration, please contact the corresponding authors via the [DigAuto Platform](https://www.digauto.org).
