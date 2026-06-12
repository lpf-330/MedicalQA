// [SM75 PATCH] Pure CUDA rmsnorm implementations for sm_75 compatibility
// Replaces flashinfer norm.cu (which requires CuTe/CUTLASS, sm_80+ only)
// Uses PyTorch native operations as fallback - no custom CUDA kernels needed

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <cmath>

#include "utils.h"

// rmsnorm: output = input * weight / sqrt(mean(input^2) + eps)
void rmsnorm(
    at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps, bool enable_pdl) {
    CHECK_INPUT(input);
    CHECK_DIM(2, input);
    auto input_sq = input.to(torch::kFloat32).pow(2);
    auto variance = input_sq.mean(/*dim=*/1, /*keepdim=*/true);
    auto input_normed = input.to(torch::kFloat32) / torch::sqrt(variance + eps);
    auto result = (input_normed * weight.to(torch::kFloat32)).to(input.scalar_type());
    output.copy_(result);
}

// fused_add_rmsnorm: residual += input; input = residual * weight / sqrt(mean(residual^2) + eps)
void sgl_fused_add_rmsnorm(
    at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps, bool enable_pdl) {
    CHECK_INPUT(input);
    CHECK_DIM(2, input);
    residual.add_(input);
    auto residual_f = residual.to(torch::kFloat32);
    auto variance = residual_f.pow(2).mean(/*dim=*/1, /*keepdim=*/true);
    auto normed = residual_f / torch::sqrt(variance + eps);
    input.copy_((normed * weight.to(torch::kFloat32)).to(input.scalar_type()));
}

// gemma_rmsnorm: output = input * (1 + weight) / sqrt(mean(input^2) + eps)
void gemma_rmsnorm(
    at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps, bool enable_pdl) {
    CHECK_INPUT(input);
    CHECK_DIM(2, input);
    auto input_f = input.to(torch::kFloat32);
    auto variance = input_f.pow(2).mean(/*dim=*/1, /*keepdim=*/true);
    auto normed = input_f / torch::sqrt(variance + eps);
    auto result = (normed * (1.0 + weight.to(torch::kFloat32))).to(input.scalar_type());
    output.copy_(result);
}

// gemma_fused_add_rmsnorm: residual += input; input = residual * (1 + weight) / sqrt(mean(residual^2) + eps)
void gemma_fused_add_rmsnorm(
    at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps, bool enable_pdl) {
    CHECK_INPUT(input);
    CHECK_DIM(2, input);
    residual.add_(input);
    auto residual_f = residual.to(torch::kFloat32);
    auto variance = residual_f.pow(2).mean(/*dim=*/1, /*keepdim=*/true);
    auto normed = residual_f / torch::sqrt(variance + eps);
    input.copy_((normed * (1.0 + weight.to(torch::kFloat32))).to(input.scalar_type()));
}
