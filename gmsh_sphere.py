import gmsh
import numpy as np
from scipy.spatial import Delaunay

# ==========================================
# Subfunctions
# ==========================================

def generate_sphere_mesh(radius=1.0, mesh_size=0.2):
    """Generates a tetrahedral mesh of a sphere using OpenCASCADE."""
    gmsh.model.add("sphere_mesh")
    gmsh.model.occ.addSphere(0, 0, 0, radius)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.model.mesh.generate(3)

def optimize_mesh():
    """Optimizes the currently active Gmsh mesh."""
    gmsh.model.mesh.optimize("Gmsh")
    gmsh.model.mesh.optimize("Netgen")

def create_gmsh_model_from_data(model_name, node_tags, node_coords, connectivity):
    """Creates a new Gmsh model from raw node and element data and returns the element tags."""
    gmsh.model.add(model_name)
    volume_tag = 1
    gmsh.model.addDiscreteEntity(3, volume_tag)
    gmsh.model.mesh.addNodes(3, volume_tag, node_tags, node_coords)

    elem_type = 4  # 4-node tetrahedron
    num_elements = len(connectivity)
    new_elem_tags = np.arange(1, num_elements + 1)
    
    # Gmsh always expects flat 1D arrays
    flat_connectivity = connectivity.flatten()
    gmsh.model.mesh.addElementsByType(volume_tag, elem_type, new_elem_tags, flat_connectivity)
    
    return new_elem_tags

def generate_and_fix_delaunay(nodes, node_tags, node_coords):
    """Runs Delaunay, loads into Gmsh to find inverted tets, fixes them, and returns final data."""
    print("\nRunning SciPy Delaunay triangulation...")
    tri = Delaunay(nodes)
    
    # Map SciPy's 0-based indices back to Gmsh's node tags
    mapped_connectivity = node_tags[tri.simplices]

    # 1. Create a temporary model just to check which elements are inverted
    elem_tags = create_gmsh_model_from_data("SciPy_Temp", node_tags, node_coords, mapped_connectivity)
    qualities = gmsh.model.mesh.getElementQualities(elem_tags, qualityName="minSICN")
    scipy_qualities = np.array(qualities)

    # 2. Fix inverted elements using the mask
    inverted_mask = scipy_qualities < 0
    print(f"Found {inverted_mask.sum()} inverted elements. Fixing them now...")

    temp = mapped_connectivity[inverted_mask, 0].copy()
    mapped_connectivity[inverted_mask, 0] = mapped_connectivity[inverted_mask, 1]
    mapped_connectivity[inverted_mask, 1] = temp
    print("All tetrahedra are now correctly right-hand oriented!")

    # 3. Create the final clean model with the fixed connectivity
    final_elem_tags = create_gmsh_model_from_data("SciPy_Fixed", node_tags, node_coords, mapped_connectivity)
    final_qualities = np.array(gmsh.model.mesh.getElementQualities(final_elem_tags, qualityName="minSICN"))

    return mapped_connectivity, final_qualities

def print_quality_stats(name, qualities):
    """Utility function to print mesh quality statistics."""
    print(f"\n--- {name} Mesh Stats ---")
    print(f"Calculated quality for {len(qualities)} elements.")
    print(f"  -> Average Quality (minSICN): {qualities.mean():.4f}")
    print(f"  -> Minimum Quality (minSICN): {qualities.min():.4f}")


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    gmsh.initialize()

    # 1. Generate and optimize the base mesh
    generate_sphere_mesh(radius=1.0, mesh_size=0.2)
    optimize_mesh()

    # Extract native Gmsh nodes and elements
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords).reshape(-1, 3)
    elem_type = 4
    elem_tags, elem_node_tags = gmsh.model.mesh.getElementsByType(elem_type)

    # Check native Gmsh qualities
    gmsh_qualities = np.array(gmsh.model.mesh.getElementQualities(elem_tags, qualityName="minSICN"))
    print_quality_stats("Native Gmsh (Optimized)", gmsh_qualities)

    # 2. Generate and fix Delaunay mesh
    fixed_connectivity, fixed_qualities = generate_and_fix_delaunay(nodes, node_tags, node_coords)
    print_quality_stats("SciPy Delaunay (Fixed)", fixed_qualities)

    # Optional: Uncomment the line below if you want the Gmsh GUI to pop up and show you the mesh
    # gmsh.fltk.run()
    gmsh.finalize()
