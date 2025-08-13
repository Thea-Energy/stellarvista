import openmc
import stellarmesh as sm

n_volumes = 8

water = openmc.Material()
water.add_element('H', 2.0, percent_type='ao')
water.add_element('O', 1.0, percent_type='ao')
water.set_density('g/cm3', 0.99821)
materials_list = [water.clone() for i in range(n_volumes)]
material_names = [f"water_{i}" for i in range(n_volumes)]
for mat, name in zip(materials_list, material_names):
    mat.name = name
materials = openmc.Materials(materials_list)

# Labrynth geometry from:
# https://grabcad.com/library/maze-rectangular-model-labyrinth-1
sm_model = sm.Geometry.from_step('labrynth.step', material_names=material_names)
sm_mesh = sm.Mesh.from_geometry(sm_model, min_mesh_size=0, max_mesh_size=50)
dagmc_model = sm.DAGMCModel.from_mesh(sm_mesh)
dagmc_model.write('dagmc.h5m')
dag_univ = openmc.DAGMCUniverse(filename="dagmc.h5m").bounded_universe(padding_distance=1)
geometry = openmc.Geometry(dag_univ)

source = openmc.IndependentSource()
source.space = openmc.stats.Point((0, 750, 0))
source.angle = openmc.stats.Isotropic()
source.energy = openmc.stats.Discrete(14.1, [1])

tallies = openmc.Tallies()
tally_mesh = openmc.RegularMesh().from_domain(geometry, dimension=[100,100,100])
mesh_filter = openmc.MeshFilter(tally_mesh)
flux_tally = openmc.Tally(name='flux')
flux_tally.filters = [mesh_filter]
flux_tally.scores = ['flux']
tallies.append(flux_tally)

settings = openmc.Settings()
settings.source = source
settings.batches = 10
settings.particles = 5000
settings.run_mode = "fixed source"

model = openmc.Model(geometry, materials, settings, tallies)
model.run()