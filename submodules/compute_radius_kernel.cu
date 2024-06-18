#include <torch/extension.h>
#include <glm/glm.hpp>

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;
	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);
	glm::mat3 M = S * R;
	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;
	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}


#define DILATE_PIXEL 0.3
// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	
	// cov[0][0] += DILATE_PIXEL;
	// cov[1][1] += DILATE_PIXEL;
	cov[0][0] = max(cov[0][0], DILATE_PIXEL);
	cov[1][1] = max(cov[1][1], DILATE_PIXEL);
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__global__ void compute_radius_cuda(int P,
	const float* orig_points,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float* projmatrix,
	const float* viewmatrix,
	const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
	float* radii)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= P)
		return;
	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	if (((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		return;
	}
    float cov3D[6];
	computeCov3D(scales[idx], rotations[idx], cov3D);
    // printf("cov3D: %d: %f %f %f %f %f %f\n", idx, cov3D[0], cov3D[1], cov3D[2], cov3D[3], cov3D[4], cov3D[5]);
	// // Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
    // printf("cov2D: %d: %f %f %f\n", idx, cov.x, cov.y, cov.z);
	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;    
	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = 3.f * sqrt(max(lambda1, lambda2));

	radii[idx] = my_radius;
    return;
}

torch::Tensor compute_radius(
	torch::Tensor& means3D,
	torch::Tensor& scales,
	torch::Tensor& rotations,
	torch::Tensor& projmatrix,
	torch::Tensor& viewmatrix,
    float focal_x, float focal_y,
    float tan_fovx, float tan_fovy
){
	const int P = means3D.size(0);
	torch::Tensor radii = torch::full({P}, 0, means3D.options());
    const int threads = 256;
    int blocks = (P + threads - 1) / threads;
	compute_radius_cuda<<<blocks, threads>>>(
		P,
		means3D.contiguous().data_ptr<float>(),
        (glm::vec3*)scales.contiguous().data_ptr<float>(),
		(glm::vec4*)rotations.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(),
        viewmatrix.contiguous().data_ptr<float>(),
        focal_x, focal_y,
        tan_fovx, tan_fovy,
		radii.contiguous().data_ptr<float>()
	);
    return radii;
}

PYBIND11_MODULE(compute_radius, m) {
    m.def("compute_radius", &compute_radius, "compute 2D radius");
}
