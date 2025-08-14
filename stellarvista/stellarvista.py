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
        and contains 'material_name' and 'volume_id' cell data.
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
            
            # Adding metadata as Cell Data
            material_names.append(volume.material)

            # Add the volume ID directly as cell data
            logger.info(f"  -> Storing Volume ID: {vol_id}")
            mesh.cell_data["DAGMC Volume ID"] = np.full(mesh.n_cells, vol_id)

            # Append the mesh for this volume to global list
            all_volume_meshes.append(mesh)

        else:
            logger.info(f"  -> Volum ID: {vol_id}, has no triangles. Skipping.")

    # Create MultiBlock
    multiblock_mesh = pv.MultiBlock(all_volume_meshes)
    # Set each block name as the material name
    for i in range(len(all_volume_meshes)):
        multiblock_mesh.set_block_name(i, material_names[i])

    logger.info("\nSuccessfully created PyVista MultiBlock dataset from pydagmc model.")

    return multiblock_mesh


def add_material_tally_data(
    mesh: pv.MultiBlock, tally_df: pd.DataFrame
) -> pv.MultiBlock:
    """Adds a MaterialFilter tally data to a PyVista MultiBlock mesh.

    This function assigns tally data from a pandas DataFrame to the
    corresponding blocks in a PyVista MultiBlock mesh based on their
    material ID.

    Args:
        mesh: A pyvista.MultiBlock object, typically generated from a geometry
        conversion. Each block is expected to have a "material" data array.
        tally_df: A pandas DataFrame containing the tally data. It must have a
        "material" column for mapping.

    Returns:
        The pyvista.MultiBlock with the newly assigned data arrays.

    Raises:
        ValueError: If a block in the mesh does not have a "material" array.
    """
    for block in mesh:
        # Check to see if material ids were assigned
        if "material" not in block.array_names:
            raise ValueError("'material' not assigned to pyvista data.")
        # Check if material id exists in tally data, if not we issue a warning and continue.
        mat_id = int(block["material"][0])
        if mat_id not in tally_df["material"].to_list():
            logger.info(
                f"Material ID {mat_id} not present in tally dataframe.", UserWarning
            )
            continue
        # Assign tally data mesh
        df_slice = tally_df.loc[tally_df["material"] == mat_id]
        for col in df_slice.columns.to_list():
            name = "".join(col)
            data = df_slice[col].to_list()
            assert len(data) == 1
            if isinstance(data[0], str):
                block.add_field_data(data[0], name)
            elif isinstance(data[0], float) or isinstance(data[0], int):
                block.point_data[name] = data * block.n_points
    return mesh


def regular_mesh_tally_to_pv(
    openmc_mesh: openmc.RegularMesh, tally_df: pd.DataFrame
) -> pv.StructuredGrid:
    """Converts OpenMC RegularMesh tally data to a PyVista StructuredGrid.

    Args:
        openmc_mesh: An OpenMC RegularMesh used for the tally.
        tally_df: A pandas DataFrame containing the tally data. Each column
        represents a tally score and each row corresponds to a mesh cell.

    Returns:
        A pyvista.StructuredGrid with corresponding tally data arrays in its
        `cell_data` attribute.
    """
    # copy grid data directly
    pv_mesh = pv.StructuredGrid(
        openmc_mesh.vertices[:, :, :, 0],
        openmc_mesh.vertices[:, :, :, 1],
        openmc_mesh.vertices[:, :, :, 2],
    )
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


def weight_windows_to_pv():

    return

