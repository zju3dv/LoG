/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/forward.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool use_filter,
	const bool debug)
{
  printf("This is RasterizeGaussiansCUDA\n");
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  // P -> primitive num
  //每个图像的pixel数量
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_point_id = torch::full({H, W}, 0, int_opts);
  torch::Tensor out_point_weight_pixel = torch::full({H, W}, 0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  // maximum rendering weight for each points
  torch::Tensor out_point_weight = torch::full({P}, 0, float_opts);
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		use_filter,
		out_color.contiguous().data<float>(),
		out_point_id.contiguous().data<int>(),
		out_point_weight_pixel.contiguous().data<float>(),
		out_point_weight.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		debug);
  }
  printf("End of is RasterizeGaussiansCUDA\n");
  return std::make_tuple(rendered, out_color, out_point_id, out_point_weight_pixel, out_point_weight, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}


torch::Tensor compute_radius(
	torch::Tensor& means3D,
	torch::Tensor& scales,
	torch::Tensor& rotations,
	torch::Tensor& viewmatrix,
	torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width
)
{ 
	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;
  
	torch::Tensor radii = torch::full({P}, 0, means3D.options());
	if(P != 0)
	{
		int M = 0;
		FORWARD::compute_radius(
			P,
			H, W,
			means3D.contiguous().data<float>(),
			scales.contiguous().data_ptr<float>(),
			rotations.contiguous().data_ptr<float>(),
			viewmatrix.contiguous().data<float>(), 
			projmatrix.contiguous().data<float>(),
			tan_fovx,
			tan_fovy,
			radii.contiguous().data<float>());
	}
	return radii;
}


torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
// std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SampleGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool use_filter,
	const bool debug)
{
  printf("Start of SampleGaussiansCUDA\n");
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  // P -> primitive num
  //每个图像的pixel数量
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_point_id = torch::full({H, W}, 0, int_opts);
  torch::Tensor out_point_weight_pixel = torch::full({H, W}, 0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  // maximum rendering weight for each points
  torch::Tensor out_point_weight = torch::full({P}, 0, float_opts);
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  //All the new sample value
  torch::Tensor range_size = torch::full({(W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y}, 0, int_opts);  // each tile has a range of primitives
  //get the index
  torch::Tensor primitive_index; // each tile related primitive ids
  torch::Tensor ranges = torch::full({(W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y,2}, 0, int_opts); // the [) range of each tile
//   torch::Tensor primitive_depth = torch::full({P}, 0, float_opts);
  torch::Tensor primitive_weight = torch::full({P}, 0, float_opts); //error 对于每个pixel都有不同的weight
//   torch::Tensor primitive_alpha = torch::full({P}, 0, float_opts);
//   torch::Tensor conic_opacity = torch::full({P, 4},0, float_opts);
//   torch::Tensor primitive_color = torch::full({P,3},0, float_opts);


  //print the shape of range_size
//   printf("range_size shape: %d, %d\n", range_size.size(0), range_size.size(1));
//   printf("range_size[0]: %d, range_size[1]: %d\n", range_size[0][0].item<int>(), range_size[1][1].item<int>());
//   printf("radii[0]: %d\n", radii[0]);
  printf("Start of rendering in SampleGaussiansCUDA\n");
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	//   rendered = CudaRasterizer::Rasterizer::forward_sample(
	//     geomFunc,
	// 	binningFunc,
	// 	imgFunc,
	//     P, degree, M,
	// 	background.contiguous().data<float>(),
	// 	W, H,
	// 	means3D.contiguous().data<float>(),
	// 	sh.contiguous().data_ptr<float>(),
	// 	colors.contiguous().data<float>(), 
	// 	opacity.contiguous().data<float>(), 
	// 	scales.contiguous().data_ptr<float>(),
	// 	scale_modifier,
	// 	rotations.contiguous().data_ptr<float>(),
	// 	cov3D_precomp.contiguous().data<float>(), 
	// 	viewmatrix.contiguous().data<float>(), 
	// 	projmatrix.contiguous().data<float>(),
	// 	campos.contiguous().data<float>(),
	// 	tan_fovx,
	// 	tan_fovy,
	// 	prefiltered,
	// 	use_filter,
	// 	out_color.contiguous().data<float>(),
	// 	out_point_id.contiguous().data<int>(),
	// 	out_point_weight_pixel.contiguous().data<float>(),
	// 	out_point_weight.contiguous().data<float>(),
	// 	radii.contiguous().data<int>(),
	// 	range_size.contiguous().data<int>(),
	// 	primitive_index,
	// 	ranges.contiguous().data<int>(),
	// 	primitive_weight,
	// 	debug);
	
	rendered = CudaRasterizer::Rasterizer::forward_sample_test(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		use_filter,
		out_color.contiguous().data<float>(),
		out_point_id.contiguous().data<int>(),
		out_point_weight_pixel.contiguous().data<float>(),
		out_point_weight.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		range_size.contiguous().data<int>(),
		ranges.contiguous().data<int>(),
		primitive_index,
		debug);
  }
	// int a=10;
//   return a;
  printf("END of SampleGaussiansCUDA\n");	
//   return std::make_tuple(rendered, out_color, out_point_id, out_point_weight_pixel, out_point_weight, radii,\
//    range_size, primitive_index, ranges);
//   return std::make_tuple(rendered, out_color, out_point_id, out_point_weight_pixel, out_point_weight, radii,\
//    range_size, ranges);
	return std::make_tuple(rendered, out_color, out_point_id, out_point_weight_pixel, out_point_weight, radii, range_size,ranges,primitive_index);
}

