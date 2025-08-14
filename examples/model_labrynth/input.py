import openmc
import stellarmesh as sm

mat = openmc.Material()
n_volumes = 8
mat.add_element('Fe', 1.0, percent_type='ao')
mat.set_density('g/cm3', 7.874)
materials_list = [mat.clone() for i in range(n_volumes)]
material_names = [f"iron_{i}" for i in range(n_volumes)]
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
settings.batches = 30
settings.particles = 1000
settings.run_mode = "fixed source"

neutron_ww = openmc.WeightWindowGenerator(tally_mesh,
                                          energy_bounds=[0.0, 0.1e6, 15e6],
                                          particle_type='neutron',
                                          method='magic',
                                          max_realizations=settings.batches,
                                          update_interval=1,
                                          on_the_fly=True)
settings.weight_window_checkpoints = {'collision': True, 'surface': True}
settings.max_history_splits = 100
settings.survival_biasing = False
settings.weight_window_generators = [neutron_ww]

model = openmc.Model(geometry, materials, settings, tallies)
model.run()