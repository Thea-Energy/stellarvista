# Stellarvista
[![Ruff Format](https://github.com/Thea-Energy/stellarvista/actions//workflows/code-quality.yml/badge.svg)](https://github.com/Thea-Energy/stellarvista/actions/workflows/code-quality.yml)

A Python package for DAGMC model inspection, tally visualization, and more for OpenMC with [PyVista](https://docs.pyvista.org/).

Stellvista allows for in-memory model inspection and tally visualization in Jupyter.
For help running PyVista in Jupyter Notebooks, see [Using PyVista in Jupyter](https://tutorial.pyvista.org/tutorial/00_jupyter/index.html).

<p align="center">
<img src="https://github.com/Thea-Energy/stellarvista/raw/main/doc/logo.png" width="100%">
</p>

## Getting Started
* Stellarvista requires an [installation of OpenMC](https://docs.openmc.org/en/stable/quickinstall.html) and PyMOAB.
* With OpenMC and PyMOAB installed, copy the stellarvista source code to your local library and install with `pip install .`
* For familiarizing yourself with PyVista, see their [examples](https://docs.pyvista.org/examples/index.html) particularly those on [plotting](https://docs.pyvista.org/examples/02-plot/).
* Stellarvista-specific example notebookds are included [here](https://github.com/Thea-Energy/stellarvista/tree/main/examples).

## Visualization Support
* DAGMC geometry and mesh
* Tallies:
    * Unstructured mesh
    * Regular mesh
    * Material filter
* Particle tracks
* Lost Particles
* Weight windows
* Point Sources
 
<p align="center">
<img src="https://github.com/Thea-Energy/stellarvista/raw/main/doc/example_collage.png" width="100%">
</p>

## Acknowledgements
Stellarvista is an open-source project by Thea Energy, who are building the world’s first planar coil stellarator.
