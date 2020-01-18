/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2019 by Contributors
 * \file rpn_target.cu
 * \brief RPN target generator
 * \author Serge Panev
 */

#include "./rpn_target-inl.h"

#include <curand.h>
#include <curand_kernel.h>

namespace mxnet {
namespace op {

#define SHMEM_SIZE 6200
#define MAX_NUM_GT 128

using namespace mshadow::cuda;

__global__ void GetGtMaxKernel(const float *box_ious,
                               float *ious_max_per_gt,
                               const int num_gt,
                               const int num_anchors) {
  __shared__ float sdata[SHMEM_SIZE];

  const int tid = threadIdx.x;

  if (blockIdx.x < num_gt) {
    float gt_max = box_ious[blockIdx.x];
    // block/num_anchors wide max reduction
    for (int curr_window = 0; curr_window < num_anchors; curr_window += SHMEM_SIZE) {
      int curr_size = min(num_anchors - curr_window, SHMEM_SIZE);
      for (int idx = tid; idx < SHMEM_SIZE; idx += blockDim.x) {
        if (idx < curr_size) {
          const int anc_idx = curr_window + idx;
          sdata[idx] = box_ious[anc_idx * num_gt + blockIdx.x];
        } else {
          sdata[idx] = 0;
        }
      }
      __syncthreads();
      curr_size = (curr_size + 1) & ~1;
      for (unsigned int offset = curr_size / 2; offset > 0; offset >>= 1) {
        for (int idx = tid; idx < offset; idx += blockDim.x) {
          sdata[idx] = fmaxf(sdata[idx], sdata[idx + offset]);
        }
        __syncthreads();
      }
      if (tid == 0) {
        gt_max = fmaxf(gt_max, sdata[0]);
      }
      __syncthreads();
    }
    if (tid == 0) {
      ious_max_per_gt[blockIdx.x] = gt_max;
    }
  }
}


__global__ void TargetSamplerKernel(const float *box_ious,
                                    const float *ious_max_per_gt,
                                    float *samples,
                                    int *matches,
                                    const float pos_iou_thresh,
                                    const float neg_iou_thresh,
                                    const int num_gt,
                                    const int num_anchors) {
  // in COCO...
  // num_anchors is bounded by 267069
  // num_gt is bounded by 90

  float cached_gt_ious_for_anc[MAX_NUM_GT];

  const int tid = threadIdx.x;

  // value based on np.spacing(np.float32(1.0))
  constexpr float eps = 1.1929e-07;

  // one thread per anchor
  // TODO: test warp-level max reduce
  for (int anc_idx = tid + blockDim.x * blockIdx.x;
       anc_idx < num_anchors;
       anc_idx += gridDim.x * blockDim.x) {
    const float *gts_for_anc = box_ious + anc_idx * num_gt;
    int max_idx = 0;
    float max_iou = gts_for_anc[0];
    int i;
    for (i = 0; i < num_gt; i++) {
      cached_gt_ious_for_anc[i] = gts_for_anc[i];
      if (cached_gt_ious_for_anc[i] > max_iou) {
        max_idx = i;
        max_iou = cached_gt_ious_for_anc[i];
      }
    }
    matches[anc_idx] = max_idx;
    for (i = 0; i < num_gt; i++) {
      if (cached_gt_ious_for_anc[i] + eps > ious_max_per_gt[i])
        break;
    }
    float sample = 0.0f;
    if (i < num_gt || max_iou >= pos_iou_thresh) {
      sample = 1.0f;
    } else if (max_iou < neg_iou_thresh && max_iou >= 0) {
      sample = -1.0f;
    }
    samples[anc_idx] = sample;
  }
}

__global__ void DisableExtraIndicesKernel(float *sample_per_anc,
                                          int *pos_samples_indices,
                                          int *neg_samples_indices,
                                          const float *rand_pool,
                                          const int num_anchors,
                                          const int max_num_pos,
                                          const int num_sample
                                          /*const int seed*/) {
  __shared__ int sdata[2];

  const int tid = threadIdx.x;

  // generating the indices to disable in tid 0 (choice)
  if (tid == 0) {
    int num_pos = 0, num_neg = 0;
    for (int idx = 0; idx < num_anchors; ++idx) {
      const float sample = sample_per_anc[idx];
      if (sample > 0) {
        pos_samples_indices[num_pos++] = idx;
      } else if (sample < 0) {
        neg_samples_indices[num_neg++] = idx;
      }
    }

    int num_pos_to_disable = num_pos - max_num_pos;
    int num_neg_to_disable = num_neg - num_sample + min(num_pos, max_num_pos);

    sdata[0] = num_pos_to_disable;
    sdata[1] = num_neg_to_disable;

    // curandState rand_state;
    // curand_init(seed, tid, 0, &rand_state);

    int rand_idx = 0;
    // Reservoir sampling
    if (num_pos_to_disable > 0) {
      for (int i = num_pos_to_disable; i < num_pos; ++i) {
        //int k = static_cast<int>(curand_uniform(&rand_state) * i);
        int k = static_cast<int>(rand_pool[rand_idx++] * i);
        if (k < num_pos_to_disable) {
          const int tmp = pos_samples_indices[k];
          pos_samples_indices[k] = pos_samples_indices[i];
          pos_samples_indices[i] = tmp;
        }
      }
    }
    if (num_neg_to_disable > 0) {
      for (int i = num_neg_to_disable; i < num_neg; ++i) {
        //int k = static_cast<int>(curand_uniform(&rand_state) * i);
        int k = static_cast<int>(rand_pool[rand_idx++] * i);
        if (k < num_neg_to_disable) {
          const int tmp = neg_samples_indices[k];
          neg_samples_indices[k] = neg_samples_indices[i];
          neg_samples_indices[i] = tmp;
        }
      }
    }
  }

  __syncthreads();

  // scatter
  int num_pos_to_disable = sdata[0];
  int num_neg_to_disable = sdata[1];

  int biggest_to_disable = max(num_pos_to_disable, num_neg_to_disable);
  for (int i = tid; i < biggest_to_disable; i += blockDim.x) {
    if (i < num_pos_to_disable) {
      sample_per_anc[pos_samples_indices[i]] = 0;
    }
    if (i < num_neg_to_disable) {
      sample_per_anc[neg_samples_indices[i]] = 0;
    }
  }
}


__device__ inline void CornerToCenter(const float4 *in, float4 *out) {
  // in global, out local mem
  const float xmin = in->x;
  const float ymin = in->x;
  const float width = in->z - xmin;
  const float height = in->w - ymin;
  out->x = xmin + width / 2;
  out->y = ymin + height / 2;
  out->z = width;
  out->w = height;
}

/*
IN:
sample_per_anc: (N) value +1 (positive), -1 (negative), 0 (ignore)
matches_indices: (N) value range [0, M)
anchors: (N, 4) encoded in corner
gt_bboxes: (M, 4) encoded in corner
OUT:
cls_targets: (N)
targets: (N, 4)
masks: (N, 4)
*/
__global__ void EncodeKernel(const float *sample_per_anc,
                             const int *matches_indices,
                             const float4 *anchors,
                             const float4 *gt_bboxes,
                             const int num_anchors,
                             float *cls_target,
                             float4 *targets,
                             float4 *masks) {
  const int tid = threadIdx.x;

  float4 g;
  float4 a;

  // one thread per anchor
  for (int anc_idx = tid + blockDim.x * blockIdx.x;
    anc_idx < num_anchors;
    anc_idx += gridDim.x * blockDim.x) {

    const float sample = sample_per_anc[anc_idx];

    float *out_cls_target = cls_target + anc_idx;
    float4 *out_target = targets + anc_idx;
    float4 *out_mask = masks + anc_idx;


    // NumPyNormalizedBoxCenterEncoder
    if (sample > 0.5) {
      const float4 *matched_bbox = gt_bboxes + matches_indices[anc_idx];
      const float4 *anc = anchors + anc_idx;
      CornerToCenter(matched_bbox, &g);
      CornerToCenter(anc, &a);
      // add box_norms? in mrcnn 1 stds and 0 means
      *out_target = make_float4((g.x - a.x) / a.z,
                                (g.y - a.y) / a.w,
                                logf(g.z / a.z),
                                logf(g.w / a.w));
      *out_mask = make_float4(1,1,1,1);
    } else {
      *out_target = make_float4(0,0,0,0);
      *out_mask = make_float4(0,0,0,0);
    }

    // SigmoidClassEncoder
    if (fabsf(sample) < 1e-5f) {
      *out_cls_target = -1.0f;
    } else {
      *out_cls_target = (sample + 1.0f) / 2.0f;
    }
  }
}

template <>
void RPNTargetRun(const RPNTargetParam& p,
                  const std::vector<TBlob> &inputs,
                  const std::vector<TBlob> &outputs,
                  const OpContext &ctx,
                  mshadow::Stream<gpu> *s) {

  auto stream = mshadow::Stream<gpu>::GetStream(s);

    // Non batched version
  const auto bboxes = inputs[rpn_target_index::kBbox].FlatToKD<gpu, 2, float>(s);
  const auto anchors = inputs[rpn_target_index::kAnchor].FlatToKD<gpu, 2, float>(s);

  auto out_cls_target = outputs[rpn_target_index::kClsTarget].FlatToKD<gpu, 1, float>(s);
  auto out_box_target = outputs[rpn_target_index::kBoxTarget].FlatToKD<gpu, 2, float>(s);
  auto out_mask_box = outputs[rpn_target_index::kBoxMask].FlatToKD<gpu, 2, float>(s);

  const int num_bbox = bboxes.shape_[0];
  const int num_anchors = anchors.shape_[0];
  const int width = p.image_size[0];
  const int height = p.image_size[1];

  constexpr int alignement = 128;

  size_t aligned_box_ious_size = align<float>(num_anchors * num_bbox, alignement);
  size_t aligned_anc_size = align<float>(num_anchors, alignement);
  size_t aligned_gt_size = align<float>(num_bbox, alignement);
  size_t aligned_anc_int_size = align<int>(num_anchors, alignement);

  int total_tmp_size = aligned_box_ious_size + aligned_anc_size * 2
                        + aligned_gt_size + aligned_anc_int_size * 3;

  mshadow::Tensor<gpu, 1, uint8_t> scratch_buffer = ctx.requested[rpn_target_index::kTempSpace]
          .get_space_typed<gpu, 1, uint8_t>(mshadow::Shape1(total_tmp_size), s);
  auto *tmp_box_ious = reinterpret_cast<float*>(scratch_buffer.dptr_);
  auto *sample_per_anc = reinterpret_cast<float*>(
                    scratch_buffer.dptr_ + aligned_box_ious_size);
  auto *tmp_gt = reinterpret_cast<float*>(
                      reinterpret_cast<uint8_t*>(sample_per_anc) + aligned_anc_size);
  auto *tmp_pos_indices = reinterpret_cast<int*>(
                            reinterpret_cast<uint8_t*>(tmp_gt) + aligned_gt_size);
  auto *tmp_neg_indices = reinterpret_cast<int*>(
                            reinterpret_cast<uint8_t*>(tmp_pos_indices) + aligned_anc_int_size);
  auto *argmax_per_anc = reinterpret_cast<int*>(
                            reinterpret_cast<uint8_t*>(tmp_neg_indices) + aligned_anc_int_size);
  auto *rand_pool_per_anc_ptr = reinterpret_cast<float*>(
                                  reinterpret_cast<uint8_t*>(argmax_per_anc) + aligned_anc_int_size);

  mshadow::Tensor<gpu, 1, float> rand_pool_per_anc(rand_pool_per_anc_ptr,
                                                   mshadow::Shape1(num_anchors));
  mshadow::Random<gpu, float> *rng = ctx.requested[rpn_target_index::kRandom]
                                       .get_random<gpu, float>(s);

  rng->SampleUniform(&rand_pool_per_anc);

  // Computing IoU
  mxnet::TShape lshape = anchors.shape_;
  mxnet::TShape rshape = bboxes.shape_;
  int lsize = lshape.ProdShape(0, lshape.ndim() - 1);
  int rsize = rshape.ProdShape(0, rshape.ndim() - 1);
  mxnet_op::Kernel<compute_iou_and_mask_invalid, gpu>::Launch(s, lsize * rsize, tmp_box_ious,
                                     anchors.dptr_, bboxes.dptr_, rsize, 0, 4,
                                     static_cast<float>(width), static_cast<float>(height));

  constexpr int NUM_BLOCKS = MAX_NUM_GT;
  constexpr int NUM_THREADS = 1024;

  GetGtMaxKernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(tmp_box_ious,
                                                         tmp_gt,
                                                         num_bbox,
                                                         num_anchors);
  MSHADOW_CUDA_POST_KERNEL_CHECK(GetGtMaxKernel);

# if 0
  cudaDeviceSynchronize();

  float tmp_gt_cpu[MAX_NUM_GT];
  cudaMemcpy(tmp_gt_cpu, tmp_gt,
    static_cast<int>(num_bbox) * sizeof (float),
    cudaMemcpyDeviceToHost);
  std::cout << "tmp_gt_cpu" << std::endl;
  for (int i = 0; i < num_bbox; ++i) {
    std::cout << tmp_gt_cpu[i] << " ";
  }
  std::cout << std::endl;
#endif

  TargetSamplerKernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(tmp_box_ious,
                                                              tmp_gt,
                                                              sample_per_anc,
                                                              argmax_per_anc,
                                                              p.pos_iou_thresh,
                                                              p.neg_iou_thresh,
                                                              num_bbox,
                                                              num_anchors);
  MSHADOW_CUDA_POST_KERNEL_CHECK(TargetSamplerKernel);

  DisableExtraIndicesKernel<<<1, NUM_THREADS, 0, stream>>>(sample_per_anc,
                                                           tmp_pos_indices,
                                                           tmp_neg_indices,
                                                           rand_pool_per_anc.dptr_,
                                                           num_anchors,
                                                           p.num_sample * p.pos_ratio,
                                                           p.num_sample);
  MSHADOW_CUDA_POST_KERNEL_CHECK(DisableExtraIndicesKernel);

  EncodeKernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(sample_per_anc,
                                                       argmax_per_anc,
                                                       reinterpret_cast<const float4*>(anchors.dptr_),
                                                       reinterpret_cast<const float4*>(bboxes.dptr_),
                                                       num_anchors,
                                                       out_cls_target.dptr_,
                                                       reinterpret_cast<float4*>(out_box_target.dptr_),
                                                       reinterpret_cast<float4*>(out_mask_box.dptr_));

  MSHADOW_CUDA_POST_KERNEL_CHECK(EncodeKernel);

/*
  cudaMemcpyAsync(out_cls_target.dptr_, sample_per_anc,
                  static_cast<int>(num_anchors) * sizeof (float),
                  cudaMemcpyDeviceToDevice,
                  stream);
*/
  cudaDeviceSynchronize();
}


DMLC_REGISTER_PARAMETER(RPNTargetParam);

NNVM_REGISTER_OP(_contrib_rpn_target)
.describe("Generate the RPN training targets from bbox and anchors.")
.set_num_inputs(2)
.set_num_outputs(3)
.set_attr_parser(ParamParser<RPNTargetParam>)
.set_attr<mxnet::FInferShape>("FInferShape", RPNTargetShape)
.set_attr<nnvm::FInferType>("FInferType", RPNTargetType)
.set_attr<FCompute>("FCompute<gpu>", RPNTargetCompute<gpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace, ResourceRequest::kRandom};
})
.add_argument("bbox", "NDArray-or-Symbol", "Ground thruth bounding box coordinates, a 3D array")
.add_argument("anchor", "NDArray-or-Symbol", "Anchors (as vbox coordinates), a 3D array")
.add_arguments(RPNTargetParam::__FIELDS__());


}  // namespace op
}  // namespace mxnet
