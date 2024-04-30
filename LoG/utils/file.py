import os
import numpy as np
import torch

def read_ply(filename):
    from plyfile import PlyData
    plydata = PlyData.read(filename)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    return positions, colors

def write_ply(outname, xyz, colors):
    """
        xyz: (N, 3)
        colors: (N, 3)
        filename: str
    """
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    from plyfile import PlyData, PlyElement
    assert xyz.shape == colors.shape
    colors = np.clip(colors, 0, 1)
    structured_array = np.zeros(xyz.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    # color => uint8
    colors = (colors * 255).astype(np.uint8)
    for i in range(xyz.shape[0]):
        structured_array[i] = tuple(xyz[i]) + tuple(colors[i])

    el = PlyElement.describe(structured_array, 'vertex')
    PlyData([el]).write(outname)

def write_mesh(outname, vertices, faces, vertex_colors):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    o3d.io.write_triangle_mesh(outname, mesh)

def read_ply_and_log(filename, scale3d=1., **kwargs):
    assert os.path.exists(filename), f'file not found: {filename}'
    if filename.endswith('.ply'):
        from plyfile import PlyData
        plydata = PlyData.read(filename)
        vertices = plydata['vertex']
        positions = scale3d * np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    elif filename.endswith('.npz'):
        plydata = dict(np.load(filename))
        positions = scale3d * plydata['xyz']
        colors = plydata['rgb'] / 255.
    elif filename.endswith('.obj'):
        import trimesh
        mesh = trimesh.load(filename)
        positions = np.asarray(mesh.vertices)
        colors = np.random.rand(*positions.shape)
    if 'offset' in kwargs:
        positions = positions - np.array(kwargs['offset']).reshape(1, 3)
    x_mean, y_mean, z_mean = positions[:, 0].mean(), positions[:, 1].mean(), positions[:, 2].mean()
    x_std, y_std, z_std = positions[:, 0].std(), positions[:, 1].std(), positions[:, 2].std()
    print(f'[{filename}] mean: {x_mean:.3f}, {y_mean:.3f}, {z_mean:.3f}')
    print(f'[{filename}] std: {x_std:.3f}, {y_std:.3f}, {z_std:.3f}')
    for sigma in [1, 2, 3]:
        flag = (positions[:, 0] > (x_mean - sigma * x_std)) & (positions[:, 0] < (x_mean + sigma * x_std)) & \
            (positions[:, 1] > (y_mean - sigma * y_std)) & (positions[:, 1] < (y_mean + sigma * y_std)) & \
            (positions[:, 2] > (z_mean - sigma * z_std)) & (positions[:, 2] < (z_mean + sigma * z_std))
        print(f'[{filename}] sigma={sigma} {flag.sum()}/{flag.shape[0]}')
        print(f'bounds: [[{x_mean-sigma*x_std:.3f}, {y_mean-sigma*y_std:.3f}, {z_mean-sigma*z_std:.3f}], [{x_mean+sigma*x_std:.3f}, {y_mean+sigma*y_std:.3f}, {z_mean+sigma*z_std:.3f}]]')
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
    print(f'[{filename}] z_min: {z_min:.3f}, z_max: {z_max:.3f}')
    return positions, colors
    
def create_from_point(filename, scale3d, ret_scale=True, **kwargs):
    if isinstance(filename, dict):
        # load from dict
        xyz = filename['xyz']
        colors = filename['colors']
    elif isinstance(filename, str) and (filename.endswith('.ply') or filename.endswith('.npz') or filename.endswith('.obj')):
        xyz, colors = read_ply_and_log(filename, scale3d, **kwargs)
    else:
        raise NotImplementedError
    print(f'[Load PLY] load from ply: {filename}')
    print(f'[Load PLY] min: {xyz.min(axis=0)}, max: {xyz.max(axis=0)}')
    xyz = torch.FloatTensor(xyz)
    colors = torch.FloatTensor(colors)
    if ret_scale:
        from simple_knn._C import distCUDA2
        dist2 = torch.clamp_min(distCUDA2(xyz.cuda()), 1e-7) #3e-4^2
        # scales = torch.clamp(torch.sqrt(dist2), self.scale_min * 2, self.scale_max / 2).to(xyz.device)
        scales = torch.sqrt(dist2).cpu()
        print(f'[Load PLY] scale: {scales.min().item():.4f}, {scales.max().item():.4f}, mean = {scales.mean().item():.4f}')
    else:
        scales = None
    return xyz, colors, scales