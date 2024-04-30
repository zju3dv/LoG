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

    M = torch.matmul(S, R)

    # // Compute 3D world covariance matrix Sigma
    Sigma = torch.matmul(M.transpose(-1, -2), M)
    return Sigma
# // Covariance is symmetric, only store upper right
#     cov3D[0] = Sigma[0][0];
#     cov3D[1] = Sigma[0][1];
#     cov3D[2] = Sigma[0][2];
#     cov3D[3] = Sigma[1][1];
#     cov3D[4] = Sigma[1][2];
#     cov3D[5] = Sigma[2][2];
# }

def transformPoint4x3(xyz, viewmatrix):
    return xyz @ viewmatrix[:3, :3] + viewmatrix[3:, :3]

# // Forward version of 2D covariance matrix computation
# def computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
def computeCov2D(cov3D, mean, viewmatrix, camera):
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
    # J = glm::mat3(
    #     focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
    #     0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
    #     0, 0, 0);
    # 
    # W = glm::mat3(
    #     viewmatrix[0], viewmatrix[4], viewmatrix[8],
    #     viewmatrix[1], viewmatrix[5], viewmatrix[9],
    #     viewmatrix[2], viewmatrix[6], viewmatrix[10]);
    W = viewmatrix[:3, :3].t()
    # 计算 T = W * J
    T = torch.matmul(W, J)
    Vrk = cov3D

    # cov = glm::transpose(T) * glm::transpose(Vrk) * T;
    cov = torch.matmul(
        torch.matmul(T.transpose(-1, -2), Vrk.transpose(-1, -2)),
        T
    )
    # // Apply low-pass filter: every Gaussian should be at least
    # // one pixel wide/high. Discard 3rd row and column.
    # cov[0][0] += DILATE_PIXEL;
    # cov[1][1] += DILATE_PIXEL;
    # return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) }
    return cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]

def compute_radius(xyz, scaling, rotation, camera):
    cov3D = computeCov3D(scaling, rotation)
    # // Compute 2D screen-space covariance matrix
    if len(camera['world_view_transform'].shape) == 3:
        viewmatrix = camera['world_view_transform'][0]
    else:
        viewmatrix = camera['world_view_transform']
    covx, covy, covz = computeCov2D(cov3D, xyz, viewmatrix, camera)
    # float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
    # // Invert covariance (EWA algorithm)
    # float det = (cov.x * cov.z - cov.y * cov.y);
    # if (det == 0.0f)
    #     return;
    det = covx * covz - covy * covy
    eps = 1e-5
    flag_zero = det.abs() < eps
    # // Compute extent in screen space (by finding eigenvalues of
    # // 2D covariance matrix). Use extent to compute a bounding rectangle
    # // of screen-space tiles that this Gaussian overlaps with. Quit if
    # // rectangle covers 0 tiles. 
    # float mid = 0.5f * (cov.x + cov.z);
    mid = 0.5 * (covx + covz)
    # float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    # float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    # lambda1 must larger than lambda 2
    lambda1 = mid + torch.sqrt(torch.clamp_min(mid * mid - det, 0.01))
    my_radius = 3. * torch.sqrt(lambda1)
    return my_radius

if __name__ == '__main__':
    rotation = torch.tensor((1, 0.1, 0, 0))[None]
    print(build_rotation(rotation))
    rotation = torch.tensor((-1, -0.1, 0, 0))[None]
    print(build_rotation(rotation))
    exit()