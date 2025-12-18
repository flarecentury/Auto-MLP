#!/usr/bin/env python3
"""
MCP-based DeepMD Inference Module
=================================

This module provides an MCP server for performing inference using DeepMD-kit potentials.
It allows AI agents to request energy and force calculations for atomic structures.

Key Features:
- Loads DeepMD models (frozen graphs).
- Calculates potential energy and atomic forces.
- Exposes calculation capabilities via MCP.

Dependencies:
- fastmcp
- deepmd-kit
- ase
- torch
"""

import os
import sys
from typing import Any, Dict, Tuple
from io import StringIO
import numpy as np
from ase.io import read
from fastmcp import FastMCP

# Try to import DeepMD, handle if missing (for documentation/structure purposes)
try:
    from deepmd.calculator import DP
    DEEPMD_AVAILABLE = True
except ImportError:
    DEEPMD_AVAILABLE = False
    print("Warning: deepmd-kit not installed. Inference features will be disabled.")

# --- Configuration ---
MODEL_PATH = os.environ.get("DEEPMD_MODEL_PATH", "models/frozen_model.pth")

# --- Helper Functions ---

def get_calculator(model_path):
    """Factory to get the DeepMD calculator."""
    if not DEEPMD_AVAILABLE:
        raise ImportError("DeepMD-kit is not available.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return DP(model=model_path)

def evaluate_structure(structure_string: str, model_path: str) -> Dict[str, Any]:
    """
    Evaluates energy and forces for a structure.
    """
    try:
        # Parse structure
        atoms = read(StringIO(structure_string), format='extxyz') # Defaulting to extxyz for now
        
        # Setup Calculator
        calc = get_calculator(model_path)
        atoms.calc = calc
        
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        return {
            "status": "success",
            "energy": energy,
            "forces": forces.tolist(),
            "natoms": len(atoms)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- MCP Server ---

mcp = FastMCP("DeepMD-Inference")

@mcp.tool()
def calculate_energy_forces(structure_string: str) -> Dict[str, Any]:
    """
    Calculates the potential energy and forces for an atomic structure using the loaded DeepMD model.
    
    Args:
        structure_string: Atomic structure in XYZ/ExtXYZ format.
    """
    return evaluate_structure(structure_string, MODEL_PATH)

@mcp.tool()
def check_model_status() -> Dict[str, Any]:
    """Checks if the DeepMD model is loaded and available."""
    return {
        "available": DEEPMD_AVAILABLE,
        "model_path": MODEL_PATH,
        "exists": os.path.exists(MODEL_PATH) if MODEL_PATH else False
    }

def main():
    mcp.run()

if __name__ == "__main__":
    main()
