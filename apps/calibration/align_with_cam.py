import argparse
import os
import numpy as np
from LoG.utils.colmap_utils import read_images_binary, read_points3d_binary, qvec2rotmat, rotmat2qvec
from LoG.utils.colmap_utils import write_points3d_binary, write_images_binary

def calculate_normal_vector(points):
    centroid = np.mean(points, axis=0)
    relative_positions = points - centroid
    covariance_matrix = np.cov(relative_positions, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]    
    return normal_vector

def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def main():
    parser = argparse.ArgumentParser(description='Align with cameras')
    parser.add_argument('--colmap_path', type=str, help='Path to the colmap sparse')
    parser.add_argument('--target_path', type=str, help='Path to the target')
    args = parser.parse_args()

    colmap_path = args.colmap_path
    target_path = args.target_path
    
    images = read_images_binary(f'{colmap_path}/images.bin')
    print(f'>> Loaded {len(images)} images')
    pt3d = read_points3d_binary(f'{colmap_path}/points3D.bin')
    print(f'>> Loaded {len(pt3d)} points3D')

    towards_list = [qvec2rotmat(v.qvec)[:, 2] for k, v in images.items()]
    towards_array = np.asarray(towards_list)

    towards_direction = towards_array.mean(0)
    towards_direction = towards_direction / np.linalg.norm(towards_direction, keepdims=True)
    print(f'>> Towards direction: {towards_direction}')

    points = []
    for k, v in images.items():
        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(v.qvec)
        w2c[:3, 3] = v.tvec
        c2w = np.linalg.inv(w2c)
        points.append(c2w[:3, 3])
    points = np.asarray(points)

    normal_vector = calculate_normal_vector(points)

    if (normal_vector * towards_direction).sum() < 0:
        normal_vector = -normal_vector

    rotation = rotation_matrix_from_vectors(normal_vector, np.array([0., 0., 1.]))

    for k, v in images.items():
        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(v.qvec)
        w2c[:3, 3] = v.tvec
        c2w = np.linalg.inv(w2c)
        c2w[:3, :3] = rotation @ c2w[:3, :3]
        c2w[:3, 3] = (rotation @ c2w[:3, 3]).T
        w2c = np.linalg.inv(c2w)
        images[k].qvec[:] = rotmat2qvec(w2c[:3, :3])
        images[k].tvec[:] = w2c[:3, 3]
    for k, v in pt3d.items():
        pt3d[k].xyz[:] = rotation @ pt3d[k].xyz
    
    os.makedirs(target_path, exist_ok=True)
    os.system(f'cp {colmap_path}/cameras.bin {target_path}')
    write_images_binary(images, f'{target_path}/images.bin')
    write_points3d_binary(pt3d, f'{target_path}/points3D.bin')

if __name__ == "__main__":
    main()