import argparse, os
import numpy as np
from LoG.utils.colmap_utils import read_images_binary, read_points3d_binary
from LoG.utils.colmap_utils import qvec2rotmat, rotmat2qvec
from LoG.utils.colmap_utils import write_images_binary, write_points3d_binary

def compute_transformation_matrix_with_scaling(source_points, target_points):
    assert source_points.shape == target_points.shape

    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target

    H = source_centered.T @ target_centered

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    scale = np.sum(S) / np.sum(source_centered ** 2)

    t = centroid_target.T - (R * scale) @ centroid_source.T

    return scale, R, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gps_path', default='./gps.npy')
    parser.add_argument('--colmap_path', default='./sparse/0/')
    parser.add_argument('--output_colmap_path', default='./sparse-align/')

    args = parser.parse_args()
    gps_path = args.gps_path
    colmap_path = args.colmap_path
    output_colmap_path = args.output_colmap_path

    os.makedirs(output_colmap_path, exist_ok=True)
    os.system(f'cp {colmap_path}{os.sep}cameras.bin {output_colmap_path}')

    gps_dict = np.load(gps_path, allow_pickle=True).tolist()
    cams = read_images_binary(f'{colmap_path}{os.sep}images.bin')
    pt3d = read_points3d_binary(f'{colmap_path}{os.sep}points3D.bin')

    cam_dict = dict()
    for v in cams.values():
        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(v.qvec)
        w2c[:3, 3] = v.tvec
        c2w = np.linalg.inv(w2c)
        cam_dict[v.name] = c2w[:3, 3]

    cam_array = []
    gps_array = []
    for k in cam_dict.keys():
        cam_array.append(cam_dict[k])
        gps_array.append(gps_dict[k])

    cam_array = np.asarray(cam_array)
    gps_array = np.asarray(gps_array)


    scale, R, t = compute_transformation_matrix_with_scaling(cam_array, gps_array)
    # set the metrci to 100meter
    scale = scale / 100
    t = t / 100
    print(f'scale: {scale}')
    print(f'R: {R}')
    print(f't: {t}')
    random_color = np.random.randint(0, 255, (cam_array.shape[0], 3))
    t[:2] = 0

    for k, v in cams.items():
        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(v.qvec)
        w2c[:3, 3] = v.tvec
        c2w = np.linalg.inv(w2c)
        c2w[:3, :3] = R @ c2w[:3, :3]
        c2w[:3, 3] = scale * (R @ c2w[:3, 3]).T + t
        w2c = np.linalg.inv(c2w)
        cams[k].qvec[:] = rotmat2qvec(w2c[:3, :3])
        cams[k].tvec[:] = w2c[:3, 3]

    for k, v in pt3d.items():
        pt3d[k].xyz[:] = scale * (R @ pt3d[k].xyz) + t

    print(f'writing {output_colmap_path}/images.bin')
    write_images_binary(cams, f'{output_colmap_path}{os.sep}images.bin')
    print(f'writing {output_colmap_path}/points3D.bin')
    write_points3d_binary(pt3d, f'{output_colmap_path}{os.sep}points3D.bin')