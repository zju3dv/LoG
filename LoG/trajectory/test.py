import numpy as np

#create a 3*4 array and a (3,) array.
def create_camera_matrix(K, R, t):
    """
    Create a camera matrix from intrinsic matrix, rotation matrix, and translation vector.
    """
    # Create a 3*4 array
    camera_matrix = np.zeros((3, 4))
    camera_matrix[:3, :3] = K
    camera_matrix[:3, 3] = t.flatten()
    return camera_matrix
