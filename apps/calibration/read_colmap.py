import os
import sys
import numpy as np
from LoG.utils.colmap_utils import read_model, qvec2rotmat

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='/home/')
    parser.add_argument('--ext', type=str, default='.bin')
    parser.add_argument('--min_views', type=int, default=3)
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    cameras, images, points3D = read_model(path=args.path, ext=args.ext)
    points3D_new = {}
    for key, val in points3D.items():
        if val.image_ids.shape[0] < args.min_views:
            continue
        points3D_new[key] = val
    print('[Read Colmap] filter {}/{} points3D, min view = {}'.format(len(points3D_new), len(points3D), args.min_views))
    points3D = points3D_new
    import cv2
    cameras_out = {}
    HW_set = set()
    for key in cameras.keys():
        p = cameras[key].params
        if cameras[key].model == 'SIMPLE_RADIAL':
            f, cx, cy, k = p
            K = np.array([f, 0, cx, 0, f, cy, 0, 0, 1]).reshape(3, 3)
            dist = np.array([[k, 0, 0, 0, 0]])
        elif cameras[key].model == 'PINHOLE':
            fx, fy, cx, cy = p
            K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
            dist = np.array([[0, 0, 0, 0, 0]])
        else:
            K = np.array([[p[0], 0, p[2], 0, p[1], p[3], 0, 0, 1]]).reshape(3, 3)
            dist = np.array([[p[4], p[5], p[6], p[7], 0.]])
        cameras_out[key] = {'K': K, 'dist': dist, 'H': cameras[key].height, 'W': cameras[key].width}
        HW_set.add((cameras[key].height, cameras[key].width))
    print('H W set', HW_set)
    mapkey = {}
    cameras_new = {}

    for key, val in images.items():
        cam = cameras_out[val.camera_id].copy()
        t = val.tvec.reshape(3, 1)
        R = qvec2rotmat(val.qvec)
        cam['R'] = R
        cam['T'] = t
        # mapkey[val.name.split('.')[0]] = val.camera_id
        cameras_new[val.name.split('.')[0]] = cam
        # cameras_new[val.name.split('.')[0].split('/')[0]] = cam
    keys = sorted(list(cameras_new.keys()))
    cameras_new = {key:cameras_new[key] for key in keys}
    print("num_cameras: {}/{}".format(len(cameras), len(cameras_new)))
    print("num_images:", len(images))
    print("num_points3D:", len(points3D))
    if len(points3D) > 0:
        keys = list(points3D.keys())
        xyz = np.stack([points3D[k].xyz for k in keys])
        rgb = np.stack([points3D[k].rgb for k in keys])
        key0 = list(cameras_new.keys())[0]
        keyn = list(cameras_new.keys())[-1]
        R0, T0 = cameras_new[key0]['R'], cameras_new[key0]['T']
        R0XT0 = xyz @ R0.T + T0.T
        RNXTN = xyz @ cameras_new[keyn]['R'].T + cameras_new[keyn]['T'].T
        if args.pca:
            mean = np.mean(xyz, axis=0)
            cov_matrix = np.cov(xyz - mean[None], rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            # sort the eigen vectors
            eigenvectors = eigenvectors[:, np.argsort(-eigenvalues)]
            eigenvectors[:, 1] *= -1
            eigenvectors[:, 2] = np.cross(eigenvectors[:, 0], eigenvectors[:, 1])
            R = eigenvectors.T
            # R[:, 2] = np.cross(R[:, 0], R[:, 1])
            print(R)
            # apply svd for R
            T = - mean[None] @ R.T
            assert (cv2.Rodrigues(cv2.Rodrigues(R)[0])[0] - R).max() < 1e-5, 'R: {}'.format(R)

            apply_trans_only = False
            if apply_trans_only:
                print('---')
                print('apply T: ', T)
                xyz_new = xyz + T
                for key, camera in cameras_new.items():
                    camera['T'] = camera['T'] - camera['R'] @ T.reshape(3, 1)
            else:
                print('---')
                print('apply R: ', R)
                print('apply T: ', T)
                xyz_new = xyz @ R.T + T
                for key, camera in cameras_new.items():
                    assert (cv2.Rodrigues(cv2.Rodrigues(camera['R'])[0])[0] - camera['R']).max() < 1e-5, 'R: {}'.format(camera['R'])
                    camera['R'] = camera['R'] @ R.T
                    R_ = cv2.Rodrigues(cv2.Rodrigues(R)[0])[0]
                    camera['T'] = camera['T'] - camera['R'] @ T.reshape(3, 1)
                    center_new = -camera['R'].T @ camera['T']
                    # camera['T'] = - camera['R'] @ center_new
            xyz = xyz_new
            R0XT0_new = xyz @ cameras_new[key0]['R'].T + cameras_new[key0]['T'].T
            RNXTN_new = xyz @ cameras_new[keyn]['R'].T + cameras_new[keyn]['T'].T
            assert (R0XT0 - R0XT0_new).max() < 1e-5, 'R0XT0: {}, RNXTN_new: {}'.format(R0XT0, R0XT0_new)
            assert (RNXTN - RNXTN_new).max() < 1e-5, 'RNXTN: {}, RNXTN_new: {}'.format(RNXTN, RNXTN_new)
        np.savez(os.path.join(args.path, 'sparse.npz'), xyz=xyz, rgb=rgb)
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb/255.)
            from os.path import join
            pcdname = join(sys.argv[1], 'sparse.ply')
            o3d.io.write_point_cloud(pcdname, pcd)
        except:
            print('---')
            print('open3d not installed')
            print('---')
        if False:
            o3d.visualization.draw_geometries([pcd])
    from LoG.dataset.camera_utils import write_camera
    write_camera(cameras_new, args.path)

if __name__ == "__main__":
    main()
