import torch
import math

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def computeCov3D(scale, rotation):
    """
        Compute 3D covariance matrix from scale and rotation
        args:
            scale: (N, 3)
            rotation: (N, 4)
    """
    # Create scaling matrix
    S = torch.diag_embed(scale)
    # // Normalize quaternion to get valid rotation
    R = build_rotation(rotation)
    L = torch.matmul(R, S)
    # // Compute 3D world covariance matrix Sigma
    Sigma = torch.matmul(L, L.transpose(-1, -2))
    return Sigma

def transformPoint4x3(xyz, viewmatrix):
    return xyz @ viewmatrix[:3, :3] + viewmatrix[3:, :3]

# // Forward version of 2D covariance matrix computation
# def computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
def computeCov2D(cov, xyz, viewmatrix, camera):
    t = transformPoint4x3(xyz, viewmatrix)
    # t = point_padding(xyz) @ camera.world_view_transform
    # t = t / t[..., 3:]
    tan_fovx = math.tan(camera['FoVx'] * 0.5)
    tan_fovy = math.tan(camera['FoVy'] * 0.5)
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[..., 0] / t[..., 2]
    tytz = t[..., 1] / t[..., 2]
    focal_y = camera['image_height'].item() / (2.0 * tan_fovy)
    focal_x = camera['image_width'].item() / (2.0 * tan_fovx)

    t[..., 0] = txtz.clip(limx).clip(None, -limx) * t[..., 2]
    t[..., 1] = tytz.clip(limy).clip(None, -limy) * t[..., 2]

    J = t.new_zeros(len(t), 3, 3)
    J[..., 0, 0] = focal_x / t[..., 2]
    J[..., 1, 1] = focal_y / t[..., 2]

    J[..., 2, 0] = -focal_x * t[..., 0] / (t[..., 2] * t[..., 2])
    J[..., 2, 1] = -focal_y * t[..., 1] / (t[..., 2] * t[..., 2])

    W = viewmatrix[..., :3, :3][None].expand(J.shape)

    T = W @ J

    VrK = t.new_zeros(len(t), 3, 3)
    VrK[..., 0, 0] = cov[..., 0]
    VrK[..., 0, 1] = cov[..., 1]
    VrK[..., 0, 2] = cov[..., 2]
    VrK[..., 1, 0] = cov[..., 1]
    VrK[..., 1, 1] = cov[..., 3]
    VrK[..., 1, 2] = cov[..., 4]
    VrK[..., 2, 0] = cov[..., 2]
    VrK[..., 2, 1] = cov[..., 4]
    VrK[..., 2, 2] = cov[..., 5]

    cov = T.mT @ VrK @ T
    cov[..., 0, 0] += 0.3
    cov[..., 1, 1] += 0.3
    return cov[..., 0, 0], cov[..., 1, 0], cov[..., 1, 1]

def computeCov2D0(cov3D, mean, viewmatrix, camera, DILATE_PIXEL=0.3):
    # // The following models the steps outlined by equations 29
    # // and 31 in "EWA Splatting" (Zwicker et al., 2002). 
    # // Additionally considers aspect / scaling of viewport.
    # // Transposes used to account for row-/column-major conventions.
    t = transformPoint4x3(mean, viewmatrix)
    tan_fovx = math.tan(camera['FoVx'] * 0.5)
    tan_fovy = math.tan(camera['FoVy'] * 0.5)
    focal_y = camera['image_height'] / (2.0 * tan_fovy)
    focal_x = camera['image_width'] / (2.0 * tan_fovx)

    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
    txtz = torch.clamp(tx / tz, -limx, limx)
    tytz = torch.clamp(ty / tz, -limy, limy)
    tx = txtz * tz
    ty = tytz * tz
    N = mean.shape[0]
    J = torch.zeros((N, 3, 3), device=mean.device)
    # 填充 J 的值
    J[:, 0, 0] = focal_x / tz
    J[:, 0, 2] = -(focal_x * tx) / (tz * tz)
    J[:, 1, 1] = focal_y / tz
    J[:, 1, 2] = -(focal_y * ty) / (tz * tz)
    W = viewmatrix[:3, :3].t()
    # 计算 T = W * J
    T = torch.matmul(J, W)
    Vrk = cov3D

    # cov = glm::transpose(T) * glm::transpose(Vrk) * T;
    cov = torch.matmul(
        T,
        torch.matmul(Vrk.transpose(-1, -2), T.transpose(-1, -2)),
    )
    # // Apply low-pass filter: every Gaussian should be at least
    # // one pixel wide/high. Discard 3rd row and column.
    cov[:, 0, 0] = torch.clamp_min(cov[:, 0, 0], DILATE_PIXEL)
    cov[:, 1, 1] = torch.clamp_min(cov[:, 1, 1], DILATE_PIXEL)
    return cov[:, 0, 0], cov[:, 1, 0], cov[:, 1, 1]

def compute_radius(xyz, scaling, rotation, camera):
    cov3D = computeCov3D(scaling, rotation)
    # // Compute 2D screen-space covariance matrix
    if len(camera['world_view_transform'].shape) == 3:
        viewmatrix = camera['world_view_transform'][0]
    else:
        viewmatrix = camera['world_view_transform']
    covx, covy, covz = computeCov2D0(cov3D, xyz, viewmatrix, camera)
    # // Invert covariance (EWA algorithm)
    det = covx * covz - covy * covy
    # // Compute extent in screen space (by finding eigenvalues of
    # // 2D covariance matrix). Use extent to compute a bounding rectangle
    # // of screen-space tiles that this Gaussian overlaps with. Quit if
    # // rectangle covers 0 tiles. 
    mid = 0.5 * (covx + covz)
    lambda1 = mid + torch.sqrt(torch.clamp_min(mid * mid - det, 0.1))
    lambda2 = mid - torch.sqrt(torch.clamp_min(mid * mid - det, 0.1))
    # lambda1 must larger than lambda 2
    my_radius = 3. * torch.sqrt(torch.maximum(lambda1, lambda2))
    return my_radius

if __name__ == '__main__':
    rotation = torch.tensor((1, 0.1, 0, 0))[None]
    print(build_rotation(rotation))
    rotation = torch.tensor((-1, -0.1, 0, 0))[None]
    print(build_rotation(rotation))
    exit()