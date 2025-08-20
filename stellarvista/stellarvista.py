import logging
import openmc
import pydagmc

import numpy as np
import pandas as pd
import pyvista as pv
import build123d as bd

logger = logging.getLogger(__name__)


def import_step(filename: str) -> pv.MultiBlock:
    """Imports a STEP file and converts it into a PyVista MultiBlock mesh.

    Args:
        filename: The path to the STEP file.

    Returns:
        A pyvista.MultiBlock object representing the geometry.
    """
    model = bd.import_step(filename)
    children = model.children
    mesh_list = []
    for part in children:
        mesh_list.append(pv.wrap(part.to_vtk_poly_data()))
    mesh = pv.MultiBlock(mesh_list)

    return mesh


def import_dagmc(filename: str) -> pv.MultiBlock:
    """Imports a DAGMC .h5m file and converts it into a PyVista MultiBlock mesh.

    The function iterates through each volume in the DAGMC model, extracts
    the triangular surface data, and creates a PyVista PolyData object for
    each volume. It then combines these into a single MultiBlock dataset.
    Material and volume ID metadata are stored as cell data.

    Args:
        filename: The path to the DAGMC .h5m file.

    Returns:
        A pyvista.MultiBlock object where each block represents a volume
        and contains 'material_name' and 'DAGMC Volume ID' cell data.
    """
    # Load the h5m file with pydagmc
    try:
        model = pydagmc.Model(filename)
    except FileNotFoundError:
        logger.info(f"Error: '{filename}' not found. Please provide a valid h5m file.")

    all_volume_meshes = []
    material_names = []

    # Iterate through each volume in the pydagmc model
    volume_ids = model.volumes_by_id.keys()
    logger.info(f"Found {len(volume_ids)} volumes in the model.")

    for vol_id in list(volume_ids):
        volume = model.volumes_by_id[vol_id]

        logger.info(f"\nProcessing Volume ID: {vol_id}")
        
        # Get the points and connectivity for the triangles in this volume
        triangle_connectivity, triangle_points = volume.get_triangle_conn_and_coords()
        
        # PyDAGMC's triangle_conn gives the indices of each face.
        # This is in a format [[a,b,c], [d,e,f], ...]
        # We need to convert this to PyVista's specific format
        # which is padded with the number of indices per face
        # ex. [3, a, b, c, 3, d, e, f, ...] 
        n_triangles = triangle_connectivity.shape[0]
        num_points_per_face = np.full((n_triangles, 1), 3, dtype=np.int64)
        # faces_array_2d is a 2D array like [[3, a, b, c], [3, d, e, f], ...]
        faces_array_2d = np.hstack((num_points_per_face, triangle_connectivity), dtype=np.int64)
        faces = faces_array_2d.flatten()

        # Create the PyVista PolyData object for this volume
        if len(triangle_points) > 0 and len(faces) > 0:
            mesh = pv.PolyData(triangle_points, faces)            
            material_names.append(volume.material)
            # Add the volume ID
            logger.info(f"  -> Storing Volume ID: {vol_id}")
            # mesh.add_field_data(vol_id, "DAGMC Volume ID")
            mesh.cell_data['DAGMC volume id'] = vol_id
            # Append the mesh for this volume to global list
            all_volume_meshes.append(mesh)

        else:
            logger.info(f"  -> Volum ID: {vol_id}, has no triangles. Skipping.")

    # Create MultiBlock
    multiblock_mesh = pv.MultiBlock(all_volume_meshes)
    # Set each block name as the material name
    for i in range(len(all_volume_meshes)):
        multiblock_mesh.set_block_name(i, material_names[i])
        all_volume_meshes[i].add_field_data(material_names[i], "material name")

    logger.info("\nSuccessfully created PyVista MultiBlock dataset from pydagmc model.")

    return multiblock_mesh


def add_material_tally_data(
    mesh: pv.MultiBlock, tally_df: pd.DataFrame, materials: openmc.Materials
) -> pv.MultiBlock:
    """Adds a MaterialFilter tally data to a PyVista MultiBlock mesh.

    This function assigns tally data from a pandas DataFrame to the
    corresponding blocks in a PyVista MultiBlock mesh based on the material name.

    Args:
        mesh: A pyvista.MultiBlock object, typically generated from a geometry
        conversion. Each block is expected to have a "material" data array.
        tally_df: A pandas DataFrame containing the tally data. It must have a
        "material" column for mapping.
        materials: The OpenMC Materials used in the simulation

    Returns:
        The pyvista.MultiBlock with the newly assigned data arrays.

    Raises:
        ValueError: If a block in the mesh does not have a "material" array.
    """
    mat_ids = [mat.id for mat in materials]
    mat_names = [mat.name for mat in materials]
    mat_id_map = {n:id for (n,id) in zip(mat_names, mat_ids)}

    # Check that the material IDs present in the dataframe are also in the materials
    for id in tally_df['material']:
        assert id in mat_ids

    for block in mesh:
        # Check if material id exists in tally data, if not we issue a warning and continue.
        mat_id = mat_id_map[block['material name']]
        if mat_id not in tally_df["material"].to_list():
            logger.info(
                f"Material ID {mat_id} not present in tally dataframe.", UserWarning
            )
            continue
        # Iterate through the columns of the DataFrame and assign data to mesh.
        # Float and Int results are added as cell_data, while strings
        # are added as field_data.
        df_slice = tally_df.loc[tally_df["material"] == mat_id]
        for col in df_slice.columns.to_list():
            name = "".join(col)
            data = df_slice[col].to_list()
            assert len(data) == 1
            if isinstance(data[0], str):
                block.add_field_data(data[0], name)
            elif isinstance(data[0], float) or isinstance(data[0], int):
                block.cell_data[name] = data * block.n_cells
    return mesh


def _openmc_regularmesh_to_pv_structured_grid(openmc_mesh: openmc.RegularMesh) -> pv.StructuredGrid:
    # copy grid data directly
    pv_mesh = pv.StructuredGrid(
        openmc_mesh.vertices[:, :, :, 0],
        openmc_mesh.vertices[:, :, :, 1],
        openmc_mesh.vertices[:, :, :, 2],
    )
    return pv_mesh


def regular_mesh_tally_to_pv(
    openmc_mesh: openmc.RegularMesh, tally_df: pd.DataFrame
) -> pv.StructuredGrid:
    """Converts OpenMC RegularMesh tally data to a PyVista StructuredGrid.

    Args:
        openmc_mesh: An OpenMC RegularMesh used in the tally.
        tally_df: A pandas DataFrame containing the tally data.

    Returns:
        A pyvista.StructuredGrid with corresponding tally data arrays in its
        `cell_data` attribute.
    """
    pv_mesh = _openmc_regularmesh_to_pv_structured_grid(openmc_mesh)
    # transfer tally data to pyvista
    for col in tally_df.columns.to_list():
        name = "".join(col)
        pv_mesh.cell_data[name] = tally_df[col]

    return pv_mesh


def unstructured_mesh_tally_to_pv(statepoint_file: str, tally_name: str) -> pv.UnstructuredGrid:
    """Loads unstructured mesh tally data and converts it to a PyVista UnstructuredGrid.

    This function reads an OpenMC statepoint file, extracts data for a
    specified unstructured mesh tally, and creates a PyVista UnstructuredGrid
    with all available score values (mean, std. dev., rel. err., sum, sum_sq)
    assigned as cell data.

    Args:
        statepoint_file: Path to the OpenMC statepoint file.
        tally_name: The name of the unstructured mesh tally to load.

    Returns:
        A pyvista.UnstructuredGrid object with the tally data. The data arrays
        are named in the format "{score} ({value_type})", e.g., "flux (mean)".
    
    Raises:
        ValueError: If the specified tally is not found or is not an
        unstructured mesh tally.
    """
    # Open the statepoint file
    sp = openmc.StatePoint(statepoint_file)
    
    tally = sp.get_tally(name=tally_name)
    
    mesh = tally.find_filter(openmc.MeshFilter).mesh

    # Initialize a dictionary to hold all the results
    tally_data = {}
    scores = tally.scores

    # List the value types we want to retrieve (all)
    value_types = ['mean', 'std_dev', 'rel_err', 'sum', 'sum_sq']

    # Get the raw tally data for the specific score and value type
    for s in scores:
        score_data_dict = {}
        for v in value_types:
            values = tally.get_values(scores=[s], value=v).squeeze(axis=1)
            score_data_dict[v] = values
        tally_data[s] = score_data_dict

    # Import unstructured mesh
    pv_mesh = pv.read(mesh.filename)

    # Add the score data to the mesh
    for s in scores:
        for v in value_types:
            scalar_data = tally_data[s][v]
            pv_mesh.cell_data[f"{s} ({v})"] = scalar_data

    return pv_mesh


def load_wws_to_pv(filename: str) -> pv.MultiBlock:
    """Loads weight window data from an OpenMC .h5 file and converts it into
    PyVista MultiBlock datasets for visualization.

    The function reads a 'weight_windows.h5' file, extracts the mesh
    geometry and weight window bounds, and creates a MultiBlock dataset for
    each set of weight windows found in the file. Each energy bin is
    stored as a separate block within the MultiBlock, with both lower and
    upper bounds assigned as cell data.

    Args:
        filename: The path to the 'weight_windows.h5' file.

    Returns:
        A pyvista MultiBlock where keys are a descriptive name for each set of
        weight windows (e.g., "neutron_ww_1") and values are PyVista
        MultiBlock objects. Each block within the MultiBlock represents
        an energy bin and contains 'Lower WW Bounds' and
        'Upper WW Bounds' as cell data.
    """
    ww_list = openmc.hdf5_to_wws(filename)
    all_ww_multiblocks = {}

    for ww in ww_list:
        ww_name = f"{ww.particle_type}_ww_{ww.id}"
        logger.info(f"Processing weight windows for: {ww_name}")
        ww_multiblock = pv.MultiBlock()

        for j in range(ww.num_energy_bins):
            # Extract data for the current energy bin
            # The data arrays are shaped (num_elements, num_energy_bins)
            current_lower_bounds = ww.lower_ww_bounds[:,:,:,j]
            current_upper_bounds = ww.upper_ww_bounds[:,:,:,j]

            if isinstance(ww.mesh, openmc.RegularMesh):
                # Create a PyVista StructuredGrid from the OpenMC RegularMesh
                pv_mesh = _openmc_regularmesh_to_pv_structured_grid(ww.mesh)
                # Reshape and add the data
                pv_mesh.cell_data["Lower WW Bounds"] = current_lower_bounds.flatten()
                pv_mesh.cell_data["Upper WW Bounds"] = current_upper_bounds.flatten()

            else:
                logger.info(
                    f"Skipping weight windows for an unsupported mesh type: "
                    f"{type(ww.mesh).__name__}."
                )
                continue

            # Add the energy bin as scalar field data
            pv_mesh.add_field_data(ww.energy_bounds[j:j+2], "Energy Bin")
            # Each mesh in the MultiBlock is named after the energy bin
            ww_multiblock.append(pv_mesh, name=f"Energy Bin {j}")
            # Add the completed MultiBlock to the main dictionary
            all_ww_multiblocks[ww_name] = ww_multiblock

    # combine all
    pv_mesh = pv.MultiBlock(all_ww_multiblocks)

    return pv_mesh


def load_particle_tracks_to_pv(filename:str) -> pv.MultiBlock:
    """Loads particle tracks from an OpenMC HDF5 file and converts them into
    a PyVista MultiBlock dataset.

    The function reads a 'tracks.h5' file, where each particle track is
    represented by its states. It extracts the spatial points and
    associated metadata for each track and creates a PyVista PolyData
    object for each primary particle. These PolyData objects are then
    aggregated into a single MultiBlock dataset. The MultiBlock's blocks
    are named with the unique identifier of the primary particle.

    The PolyData objects contain a continuous series of lines representing
    the particle trajectories, with all metadata (energy, time, weight,
    cell ID, cell instance, and material ID) stored as point data for
    easy plotting and analysis.

    Args:
      filename: The path to the 'tracks.h5' file.

    Returns:
      A pyvista.MultiBlock object where each block represents a primary
      particle's complete track (including all its secondary particle
      histories). Each block is a `pyvista.PolyData` object. The `point_data`
      for each `PolyData` object includes the following arrays:
      - 'E': The energy of the particle at each state.
      - 'time': The time of the particle at each state.
      - 'wgt': The weight of the particle at each state.
      - 'cell_id': The geometry cell ID at each state.
      - 'cell_instance': The cell instance at each state.
      - 'material_id': The material ID at each state.
      - 'particle_type': A mapped integer for the particle's type, 
                        {neutron: 0, photon: 1, electron: 2, positron: 3}.
    """
    tracks = openmc.Tracks(filename)
    state_names =  ['E', 'time', 'wgt', 'cell_id', 'cell_instance', 'material_id']
    particle_data = {'state_data': {state:[] for state in state_names},
                    'current_point_count': 0,
                    'points': [],
                    'line_connectivity': [],
                    'particle_types': [],}

    multiblock = pv.MultiBlock()

    for primary_particle in tracks:
        for track in primary_particle.particle_tracks:
            # Get the points for the current track
            track_points_structured = track.states['r']
            num_track_points = track_points_structured.shape[0]
            
            # Convert the structured array to a standard NumPy array for PyVista
            track_points = np.empty((num_track_points, 3), dtype=np.float64)
            track_points[:, 0] = track_points_structured['x']
            track_points[:, 1] = track_points_structured['y']
            track_points[:, 2] = track_points_structured['z']

            # Append to the global list of points
            particle_data['points'].append(track_points)
            
            # Build the line connectivity
            particle_data['line_connectivity'].append(num_track_points)
            particle_data['line_connectivity'].extend(range(particle_data['current_point_count'],
                                                            particle_data['current_point_count']
                                                            + num_track_points))
            
            # Store metadata for each point
            particle_data['particle_types'].extend([track.particle] * num_track_points)
            for s in state_names:
                particle_data['state_data'][s].extend(track.states[s])
            
            particle_data['current_point_count'] += num_track_points

        # Create the PyVista PolyData object from the points and lines
        combined_points = np.vstack(particle_data['points'])
        mesh = pv.PolyData(combined_points, lines=particle_data['line_connectivity'])

        # Convert particle type strings to an integer array
        unique_particle_types = list(set(particle_data['particle_types']))
        particle_map = {name: i for i, name in enumerate(unique_particle_types)}
        mapped_particle_types = np.array([particle_map[p] for p in particle_data['particle_types']])
        mesh.point_data['particle_type'] = mapped_particle_types
        # Add a field data array to store the mapping from integer to particle name
        mesh.add_field_data(str(particle_map), "particle_type_map")

        # Add scalar data to the mesh
        for s in state_names:
            mesh.point_data[s] = np.array(particle_data['state_data'][s])

        multiblock.append(mesh)

    for i in range(len(multiblock)):
        multiblock.set_block_name(i, f"Track {tracks[i].identifier}")
    
    return multiblock