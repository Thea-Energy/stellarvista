import logging
import openmc
import pydagmc

import numpy as np
import pandas as pd
import pyvista as pv
import build123d as bd

logger = logging.getLogger(__name__)

def import_step(filename: str) -> pv.MultiBlock:
    model = bd.import_step(filename)
    children = model.children
    mesh_list = []
    for part in children:
        mesh_list.append(pv.wrap(part.to_vtk_poly_data()))
    mesh = pv.MultiBlock(mesh_list)

    return mesh


def import_h5m(filename: str) -> pv.MultiBlock:
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
            material_tag = volume.material
            material_names.append(material_tag)
            if material_tag:
                logger.info(f"  -> Found material: {material_tag}")
                mesh.add_field_data([material_tag], "material_name")

            # Add the volume ID directly as cell data
            logger.info(f"  -> Storing Volume ID: {vol_id}")
            mesh["volume_id"] = np.full(mesh.n_cells, vol_id)

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
    """Add MaterialFilter tally data to a pyvista MultiBlock mesh.

    Args:
        mesh: A pyvista MultiBlock object, typically generated with `bd_solid_to_pv_mesh`.
        tally_df: A pandas DataFrame containing the tally data.

    Returns:
        The pyvista MultiBlock with newly assigned data arrays.
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
    """Convert OpenMC RegularMesh tally data to a Pyista StructuredGrid.

    Args:
        openmc_mesh: An OpenMC RegularMesh used for the tally.
        tally_df: A pandas DataFrame containing the tally data.

    Returns:
        A pyvista StructuredGrid with corresponding tally data arrays.
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