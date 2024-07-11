import numpy as np
import scipy.sparse as sparse


def compute_faces_areas(vertices, faces):
    """
    Compute per-face areas of a triangular mesh

    Parameters
    -----------------------------
    vertices : (n,3) array of vertices coordinates
    faces    : (m,3) array of vertex indices defining faces

    Output
    -----------------------------
    faces_areas : (m,) array of per-face areas
    """

    v1 = vertices[faces[:,0]]  # (m,3)
    v2 = vertices[faces[:,1]]  # (m,3)
    v3 = vertices[faces[:,2]]  # (m,3)
    faces_areas = 0.5 * np.linalg.norm(np.cross(v2-v1,v3-v1),axis=1)  # (m,)

    return faces_areas


def compute_vertex_areas(vertices, faces, faces_areas=None):
    """
    Compute per-vertex areas of a triangular mesh.
    Area of a vertex, approximated as one third of the sum of the area of its adjacent triangles.

    Parameters
    -----------------------------
    vertices    : (n,3) array of vertices coordinates
    faces       : (m,3) array of vertex indices defining faces
    faces_areas :

    Output
    -----------------------------
    vert_areas : (n,) array of per-vertex areas
    """
    N = vertices.shape[0]

    if faces_areas is None:
        faces_areas = compute_faces_areas(vertices,faces)  # (m,)

    I = np.concatenate([faces[:,0], faces[:,1], faces[:,2]])
    J = np.zeros_like(I)

    V = np.tile(faces_areas / 3, 3)

    # Get the (n,) array of vertex areas
    vertex_areas = np.array(sparse.coo_matrix((V, (I, J)), shape=(N, 1)).todense()).flatten()

    return vertex_areas

def compute_vertex_normals(vertices, faces):
    """
    Compute per-vertex normals of a triangular mesh, weighted by the area of adjacent faces.

    Parameters
    -----------------------------
    vertices     : (n,3) array of vertices coordinates
    faces        : (m,3) array of vertex indices defining faces

    Output
    -----------------------------
    vert_areas : (n,) array of per-vertex areas
    """

    n_faces = faces.shape[0]
    n_vertices = vertices.shape[0]

    v1 = vertices[faces[:, 0]]  # (m,3)
    v2 = vertices[faces[:, 1]]  # (m,3)
    v3 = vertices[faces[:, 2]]  # (m,3)

    # That is 2* A(T) n(T) with A(T) area of face T
    face_normals_weighted = np.cross(1e3*(v2-v1), 1e3*(v3-v1))  # (m,3)

    # A simple version should be :
    # vert_normals = np.zeros((n_vertices,3))
    # np.add.at(vert_normals, faces.flatten(),np.repeat(face_normals_weighted,3,axis=0))
    # But this code is way faster in practice

    In = np.repeat(faces.flatten(), 3)  # (9m,)
    Jn = np.tile(np.arange(3), 3*n_faces)  # (9m,)
    Vn = np.tile(face_normals_weighted, (1,3)).flatten()  # (9m,)

    vert_normals = sparse.coo_matrix((Vn, (In, Jn)), shape=(n_vertices, 3))
    vert_normals = np.asarray(vert_normals.todense())
    vert_normals /= (1e-6 + np.linalg.norm(vert_normals, axis=1, keepdims=True))

    return vert_normals.astype(np.float32)

