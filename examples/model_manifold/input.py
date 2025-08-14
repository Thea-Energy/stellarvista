import openmc

# Example adapted from the steel pip manifold example here:
# https://docs.openmc.org/en/v0.12.1/examples/unstructured-mesh-part-ii.html

from IPython.display import display, clear_output
import urllib.request

manifold_geom_url = 'https://tinyurl.com/rp7grox' # 99 MB
manifold_mesh_url = 'https://tinyurl.com/wojemuh' # 5.4 MB

def download(url, filename):
    """
    Helper function for retrieving dagmc models
    """
    def progress_hook(count, block_size, total_size):
        prog_percent = 100 * count * block_size / total_size
        prog_percent = min(100., prog_percent)
        clear_output(wait=True)
        display('Downloading {}: {:.1f}%'.format(filename, prog_percent))

    urllib.request.urlretrieve(url, filename, progress_hook)

# get the manifold DAGMC geometry file
download(manifold_geom_url, 'dagmc.h5m')
# get the manifold tet mesh
download(manifold_mesh_url, 'unstructured_mesh.h5m')

# Materials
air = openmc.Material(name='air')
air.set_density('g/cc', 0.001205)
air.add_element('N', 0.784431)
air.add_element('O', 0.210748)
air.add_element('Ar',0.0046)

steel = openmc.Material(name='steel')
steel.set_density('g/cc', 8.0)
steel.add_element('Si', 0.010048)
steel.add_element('S', 0.00023)
steel.add_element('Fe', 0.669)
steel.add_element('Ni', 0.12)
steel.add_element('Mo', 0.025)
steel.add_nuclide('P31',0.00023)
steel.add_nuclide('Mn55',0.011014)

materials = openmc.Materials([air, steel])

# Geometry
dag_univ = openmc.DAGMCUniverse(filename="dagmc.h5m").bounded_universe()
geometry = openmc.Geometry(dag_univ)

# Source
src_pnt = openmc.stats.Point(xyz=(0.0, 0.0, 0.0))
src_energy = openmc.stats.Discrete(x=[5.e+06], p=[1.0])
source = openmc.Source(space=src_pnt, energy=src_energy)

# Tallies
unstructured_mesh = openmc.UnstructuredMesh("unstructured_mesh.h5m", library='moab')
mesh_filter = openmc.MeshFilter(unstructured_mesh)

tally = openmc.Tally(name='flux')
tally.filters = [mesh_filter]
tally.scores = ['flux']
tally.estimator = 'tracklength'
tallies = openmc.Tallies([tally])

# Settings
settings = openmc.Settings()
settings.source = source
settings.run_mode = "fixed source"
settings.batches = 200
settings.particles = 5000

model = openmc.Model(geometry, materials, settings, tallies)
model.run()