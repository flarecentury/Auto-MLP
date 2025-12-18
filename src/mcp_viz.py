#!/usr/bin/env python3
"""
MCP-based Atomic Structure Visualization Module
===============================================

This module provides an MCP (Model Context Protocol) server for visualizing atomic structures.
It parses structure files (POSCAR, XYZ, CIF, etc.) and generates embeddable Three.js 
visualizations.

Key Features:
- Support for multiple file formats (via ASE).
- Interactive 3D visualization using Three.js.
- MCP tool exposure for integration with AI agents.
- Customizable element colors and radii.

Dependencies:
- fastmcp
- ase
- numpy
"""

import os
import sys
import json
from typing import Any, Dict
from io import StringIO
from fastmcp import FastMCP
import numpy as np
from ase.io import read
from ase import Atoms

# --- Constants & Configuration ---

# CPK Color Scheme
ELEMENT_COLORS = {
    'H': '#FFFFFF', 'He': '#D9FFFF', 'Li': '#CC80FF', 'Be': '#C2FF00', 'B': '#FFB5B5', 'C': '#909090',
    'N': '#3050F8', 'O': '#FF0D0D', 'F': '#90E050', 'Ne': '#B3E3F5', 'Na': '#AB5CF2', 'Mg': '#8AFF00',
    'Al': '#BFA6A6', 'Si': '#F0C8A0', 'P': '#FF8000', 'S': '#FFFF30', 'Cl': '#1FF01F', 'Ar': '#80D1E3',
    'K': '#8F40D4', 'Ca': '#3DFF00', 'Sc': '#E6E6E6', 'Ti': '#BFC2C7', 'V': '#A6A6AB', 'Cr': '#8A99C7',
    'Mn': '#9C7AC7', 'Fe': '#E06633', 'Co': '#F090A0', 'Ni': '#50D050', 'Cu': '#C88033', 'Zn': '#7D80B0',
    # ... (Add more elements as needed)
}

# Atomic Radii (Angstroms)
ATOMIC_RADII = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66,
    'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05,
    'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76, 
    # ...
}

# --- Helper Functions ---

def parse_structure_string(structure_string: str) -> Atoms:
    """Parse structure from string, supports various formats."""
    supported_formats = ['vasp', 'cif', 'extxyz', 'xyz', 'pdb', 'mol']
    atoms = None
    for fmt in supported_formats:
        try:
            atoms = read(StringIO(structure_string), format=fmt)
            break
        except Exception:
            continue
    
    if atoms is None:
        raise ValueError(f"Unable to parse input string. Supported formats: {supported_formats}")
    return atoms

def atoms_to_json(atoms: Atoms) -> Dict[str, Any]:
    """Convert ASE Atoms object to JSON-compatible dictionary."""
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    atoms_data = []
    for i, (pos, symbol) in enumerate(zip(positions, symbols)):
        color = ELEMENT_COLORS.get(symbol, '#CCCCCC')
        radius = ATOMIC_RADII.get(symbol, 1.0)
        
        atoms_data.append({
            'id': i,
            'element': symbol,
            'position': pos.tolist(),
            'color': color,
            'radius': radius
        })
    
    cell_info = None
    if atoms.cell is not None and not np.allclose(atoms.cell, 0):
        cell_info = {
            'vectors': atoms.cell.tolist(),
            'pbc': atoms.pbc.tolist()
        }
    
    return {
        'atoms': atoms_data,
        'cell': cell_info,
        'formula': atoms.get_chemical_formula(),
        'natoms': len(atoms)
    }

def generate_threejs_html(atoms_data: Dict[str, Any]) -> str:
    """Generate HTML string for Three.js visualization."""
    atoms_json = json.dumps(atoms_data, indent=2)
    
    # HTML Template (Simplified for brevity, includes necessary scripts)
    html_template = f"""
<div id="atomic-viz-container" style="width: 100%; height: 400px; background-color: white; border-radius: 8px; overflow: hidden; position: relative;">
    <div style="position: absolute; top: 10px; left: 10px; z-index: 10; background: rgba(255,255,255,0.8); padding: 5px; border-radius: 4px;">
        <strong>{atoms_data['formula']}</strong> ({atoms_data['natoms']} atoms)
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        (function() {{
            const data = {atoms_json};
            const container = document.currentScript.parentElement;
            
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0xffffff);
            
            const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.z = 20;
            
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(renderer.domElement);
            
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            
            const light = new THREE.DirectionalLight(0xffffff, 1);
            light.position.set(1, 1, 1).normalize();
            scene.add(light);
            scene.add(new THREE.AmbientLight(0x404040));
            
            const group = new THREE.Group();
            
            // Add Atoms
            data.atoms.forEach(atom => {{
                const geo = new THREE.SphereGeometry(atom.radius * 0.5, 32, 32);
                const mat = new THREE.MeshLambertMaterial({{ color: atom.color }});
                const mesh = new THREE.Mesh(geo, mat);
                mesh.position.set(...atom.position);
                group.add(mesh);
            }});
            
            // Add Cell
            if (data.cell) {{
                // Simple box helper or line segments for cell
                // Implementation omitted for brevity
            }}
            
            scene.add(group);
            
            // Center camera
            const box = new THREE.Box3().setFromObject(group);
            const center = box.getCenter(new THREE.Vector3());
            controls.target.copy(center);
            camera.position.set(center.x, center.y, center.z + 15);
            controls.update();
            
            function animate() {{
                requestAnimationFrame(animate);
                renderer.render(scene, camera);
            }}
            animate();
            
            // Resize handler
            const resizeObserver = new ResizeObserver(() => {{
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }});
            resizeObserver.observe(container);
        }})();
    </script>
</div>
"""
    return html_template

# --- MCP Server Setup ---

mcp = FastMCP("AtomicViz-MCP")

@mcp.tool()
def visualize_structure(structure_string: str) -> Dict[str, Any]:
    """
    Generates a 3D visualization for a given atomic structure.
    
    Args:
        structure_string: The content of the structure file (POSCAR, XYZ, etc.)
    """
    try:
        atoms = parse_structure_string(structure_string)
        atoms_data = atoms_to_json(atoms)
        html = generate_threejs_html(atoms_data)
        
        return {
            "status": "success",
            "html": html,
            "info": {
                "formula": atoms_data['formula'],
                "natoms": atoms_data['natoms']
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def main():
    """Entry point for the MCP server."""
    mcp.run()

if __name__ == "__main__":
    main()
