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
 * \file mrcnn_mask_target.cu
 * \brief Mask-RCNN target generator
 * \author Serge Panev
 */

#include "./mrcnn_mask_target-inl.h"
#include <cub/cub.cuh>

namespace mxnet {
namespace op {

using namespace mshadow::cuda;

// The maximum number of blocks to use in the default kernel call.
constexpr int MAXIMUM_NUM_BLOCKS = 4096;

inline int CUDA_GET_BLOCKS(const int N) {
  return std::max(
      std::min(
          (N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock,
          MAXIMUM_NUM_BLOCKS),
      // Use at least 1 block, since CUDA does not allow empty block.
      1);
}

// Kernels

/*rle cuda kernels are cuda version of the corresponding cpu functions here
https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c
these are only a subset of rle kernels.*/

typedef unsigned int uint;

template <typename DType>
__global__ void mask_class_encode(const DType *matches,
                                  const DType *cls_targets,
                                  DType* mask_cls,
                                  int batch_size,
                                  int num_classes,
                                  int num_rois,
                                  int mask_size_h,
                                  int mask_size_w) {
  const int mask_idx = blockIdx.x;
  const int cls_idx = mask_idx % num_classes;
  const int roi_idx = (mask_idx / num_classes) % num_rois;
  const int batch_idx = (mask_idx / num_classes / num_rois);

  const int mask_vol = mask_size_h * mask_size_w;

  DType* mask_cls_out = mask_cls + mask_idx * mask_vol;

  DType cls_target = cls_targets[batch_idx * num_rois + roi_idx];
  DType out_val = (cls_target == cls_idx);
  for (int mask_pixel = threadIdx.x; mask_pixel < mask_vol; mask_pixel += blockDim.x) {
    mask_cls_out[mask_pixel] = out_val;
  }
}

template <typename xpu, typename DType>
mshadow::Tensor<xpu, 1, DType> get_1d_tensor(int size,
                                             mshadow::Stream<xpu> *stream,
                                             const OpContext &ctx) {
  return ctx.requested[mrcnn_index::kTempSpace]
          .get_space_typed<xpu, 1, DType>(mshadow::Shape1(size), stream);
}

constexpr int NTHREADS = 256;
constexpr int NEDGES = NTHREADS;
constexpr int THREADS_PER_WARP = 32;

__device__ inline int binary_search(double x, double* table, const int N) {
  int min_idx = 0;
  int max_idx = N;
  int mid_idx = N / 2;
  /*if (blockIdx.x == 0 && threadIdx.x == 32) {*/
    /*printf("HAHA %f\n", x);*/
  /*}*/
  while(max_idx > min_idx){
      if(x > table[mid_idx]) {
          min_idx = mid_idx+1;
  /*if (blockIdx.x == 0 && threadIdx.x == 32) {*/
    /*printf("HOHO %d %d %d\n", min_idx, mid_idx, max_idx);*/
  /*}*/
      }
      else if (x < table[mid_idx]) {
          max_idx = mid_idx;
  /*if (blockIdx.x == 0 && threadIdx.x == 32) {*/
    /*printf("HIHI %d %d %d\n", min_idx, mid_idx, max_idx);*/
  /*}*/
      }
      else {
          mid_idx++;
          break;
      }
      mid_idx = (min_idx + max_idx) / 2;
  }
  /*if (blockIdx.x == 0 && threadIdx.x == 32) {*/
    /*printf("HLEHLE %d %d %d\n", min_idx, mid_idx, max_idx);*/
  /*}*/
  return mid_idx;
}

template <typename DType>
__device__ inline double2 crop_and_scale(double2 xy, double ratio_w, double ratio_h, const DType* roi) {
  double x = ratio_w * (xy.x - (double)roi[0]);
  double y = ratio_h * (xy.y - (double)roi[1]);
  return {x, y};
}

template <typename DType>
__global__ void rasterize_kernel(const DType* matches,
                                 const int* gt_mask_meta,
                                 const double* gt_mask_coordinates,
                                 const DType* roi_data,
                                 const int num_of_polygons,
                                 const long h,
                                 const long w,
                                 unsigned char* mask) {
  const int roi_idx = blockIdx.x;
  const int tid = threadIdx.x;

  const int gt_mask_idx = static_cast<int>(matches[roi_idx]);

  int current_poly_idx;
  for (current_poly_idx = 0; current_poly_idx < num_of_polygons; ++current_poly_idx) {
    if (gt_mask_meta[current_poly_idx * 3] == gt_mask_idx) break;
  }

  // if (tid == 0) {
  //   printf("gt_mask_idx=%d-current_poly_idx=%d\n", gt_mask_idx, current_poly_idx);
  // }

  const DType *roi = roi_data + (roi_idx * 4);
  DType roi_w = fmaxf(roi[2] - roi[0], 1.0f);
  DType roi_h = fmaxf(roi[3] - roi[1], 1.0f);
  const double ratio_w = (double)w / (double)roi_w;
  const double ratio_h = (double)h / (double)roi_h;

  __shared__ double2 vertices[NEDGES];
  typedef cub::BlockRadixSort<double, NTHREADS, 1> cub_sort;
  constexpr int NWARPS = NTHREADS / THREADS_PER_WARP;
  __shared__ union {
    double intersections[NEDGES * NWARPS];
    typename cub_sort::TempStorage temp_storage;
  } scratch;

  double temp_intersections[NWARPS];

  unsigned char* const mask_ptr = mask + roi_idx * h * w;

  const int warp_id = tid / THREADS_PER_WARP;
  const int lane_id = tid % THREADS_PER_WARP;
  const double invalid_intersection = 2 * w;
  const long aligned_h = ((h + NWARPS - 1) / NWARPS) * NWARPS;

  for (int current_y = warp_id;
    current_y < h;
    current_y += NWARPS) {
    if (current_y < h) {
      for (int x = lane_id; x < w; x += THREADS_PER_WARP) {
        mask_ptr[x + current_y * w] = 0;
      }
    }
  }
  __syncthreads();

  for (;current_poly_idx < num_of_polygons && gt_mask_meta[current_poly_idx * 3] == gt_mask_idx;
       current_poly_idx++) {
    // current_poly_meta (mask_idx, offset, count)
    const int* current_poly_meta = gt_mask_meta + current_poly_idx * 3;
    const double2 *xy = reinterpret_cast<const double2*>(gt_mask_coordinates + current_poly_meta[1]);
    const int k = current_poly_meta[2] / 2;

    int current_k = 0;

    for (int current_k_offset = 0; current_k_offset < k; current_k_offset += NEDGES) {
      current_k = min((k - current_k_offset), (int)NEDGES);

      if (tid < current_k) {
        int v_idx = current_k_offset + tid;
        vertices[tid] = crop_and_scale(xy[v_idx], ratio_w, ratio_h, roi);
      }

      __syncthreads();

      for (int current_y = warp_id;
          current_y < aligned_h;
          current_y += NWARPS) {
        double my_y = current_y + 0.5;
        for (int edge = lane_id; edge < NEDGES; edge += THREADS_PER_WARP) {
          scratch.intersections[warp_id * NEDGES + edge] = invalid_intersection;
        }
        __syncthreads();
        if (current_y < h) {
          for (int edge = lane_id; edge < current_k; edge += THREADS_PER_WARP) {
            const int previous_edge = (edge - 1 + current_k) % current_k;
            const double2 vert1 = vertices[previous_edge];
            const double2 vert2 = vertices[edge];
            if (vert1.x == vert2.x) {
              double min_y = fmin(vert1.y, vert2.y);
              double max_y = fmax(vert1.y, vert2.y);
              if (my_y <= max_y && my_y >= min_y) {
                scratch.intersections[warp_id * NEDGES + edge] = vert1.x;
              }
            } else if (vert1.y != vert2.y) {
              double my_intersection = (my_y * (vert1.x - vert2.x) +
                                        vert2.x * vert1.y -
                                        vert1.x * vert2.y) /
                                      (vert1.y - vert2.y);
              if (my_intersection <= fmax(vert1.x, vert2.x) &&
                  my_intersection >= fmin(vert1.x, vert2.x)) {
                scratch.intersections[warp_id * NEDGES + edge] = my_intersection;
              }
            }
          }
        }

        __syncthreads();

  #pragma unroll
        for (int i = 0; i < NWARPS; ++i) {
          temp_intersections[i] = scratch.intersections[i * NEDGES + tid];
        }

        __syncthreads();

  #pragma unroll
        for (int i = 0; i < NWARPS; ++i) {
          double temp[1] = { temp_intersections[i] };
          cub_sort(scratch.temp_storage).Sort(temp);
          temp_intersections[i] = temp[0];
        }

        __syncthreads();

  #pragma unroll
        for (int i = 0; i < NWARPS; ++i) {
          scratch.intersections[i * NEDGES + tid] = temp_intersections[i];
        }

        __syncthreads();

        if (current_y < h) {
          for (int x = lane_id; x < w; x += THREADS_PER_WARP) {
            const double my_x = x + 0.5;
            const int place = binary_search(my_x, scratch.intersections + NEDGES * warp_id, NEDGES);
            mask_ptr[x + current_y * w] = (mask_ptr[x + current_y * w] + place) % 2;
          }
        }
        __syncthreads();
      }  //  for current_y
      __syncthreads();
    }  // for k
    __syncthreads();
  }  // for polygons
}

template<typename DType>
__global__ void write_masks_to_output_kernel(unsigned char *masks_in, DType *masks_out, const int mask_size,
                                             const DType *cls_targets, int num_classes) {
  const int roi_idx = blockIdx.x / num_classes;
  const int class_idx = blockIdx.x % num_classes;
  const int tid = threadIdx.x;

  const int in_mask_offset = roi_idx * mask_size * mask_size;
  const int out_mask_offset = (roi_idx * num_classes + class_idx) * mask_size * mask_size;

  for(int j = tid; j < mask_size * mask_size; j += blockDim.x) {
    masks_out[out_mask_offset + j] = (DType)masks_in[in_mask_offset + j];
  }
}

template<>
void MRCNNMaskTargetRun<gpu>(const MRCNNMaskTargetParam& param, const std::vector<TBlob> &inputs,
                             const std::vector<TBlob> &outputs, const OpContext &ctx,
                             mshadow::Stream<gpu> *s) {
  using namespace mxnet_op;
  using mshadow::Tensor;

  // only works with square mask targets
  const int M = param.mask_size[0];

  MSHADOW_REAL_TYPE_SWITCH(inputs[mrcnn_index::kRoi].type_flag_, DType, {
    const auto rois = inputs[mrcnn_index::kRoi].FlatToKD<gpu, 3, DType>(s);
    const auto matches = inputs[mrcnn_index::kMatches].FlatTo2D<gpu, DType>(s);
    const auto cls_targets = inputs[mrcnn_index::kClasses].FlatTo2D<gpu, DType>(s);
    const auto gt_mask_meta = inputs[mrcnn_index::kMasksMeta].FlatTo2D<gpu, int>(s);
    const auto gt_mask_coordinates = inputs[mrcnn_index::kMasksCoords].FlatTo2D<gpu, double>(s);

    auto out_masks = outputs[mrcnn_index::kMask].FlatToKD<gpu, 5, DType>(s);
    auto out_mask_cls = outputs[mrcnn_index::kMaskClasses].FlatToKD<gpu, 5, DType>(s);

    int batch_size = rois.shape_[0];

    // Mask Target generation
    int num_of_rois = rois.shape_[1];

    auto d_mask_t = get_1d_tensor<gpu, unsigned char>(M * M * batch_size * num_of_rois, s, ctx);

    auto stream = mshadow::Stream<gpu>::GetStream(s);

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);

    rasterize_kernel<<<batch_size * num_of_rois, NEDGES, 0, stream>>>(matches.dptr_,
                                                                      gt_mask_meta.dptr_,
                                                                      gt_mask_coordinates.dptr_,
                                                                      rois.dptr_,
                                                                      gt_mask_meta.shape_[0],
                                                                      M, M,
                                                                      d_mask_t.dptr_);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float ms = 0;
    // cudaEventElapsedTime(&ms, start, stop);
    // std::cout << "rasterize_kernel time: " <<  milliseconds << "ms" << std::endl;
#if 0
    for (int mask_idx = 0; mask_idx < num_of_rois / 4; ++mask_idx) {
      unsigned char* test = new unsigned char[28 * 28];
      cudaMemcpy(test, d_mask_t.dptr_ + 28 * 28 * mask_idx,
                 28 * 28 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
      std::cout << "Mask #" << mask_idx << std::endl;
      for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
          if (test[i*28 + j] == 0) {
            std::cout << "  ";
          } else {
            std::cout << "##";
          }
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
#endif
    // out: 2 * (B, N, C, MS, MS)
    const int total_num_of_masks = batch_size * num_of_rois * param.num_classes;
    write_masks_to_output_kernel<<<total_num_of_masks, 256, 0, stream>>>(d_mask_t.dptr_,
                                                                         out_masks.dptr_, M,
                                                                         cls_targets.dptr_,
                                                                         param.num_classes);


    mask_class_encode<<<total_num_of_masks, 1024, 0, stream>>> (matches.dptr_, cls_targets.dptr_,
                                                                out_mask_cls.dptr_,
                                                                batch_size,
                                                                param.num_classes,
                                                                num_of_rois,
                                                                param.mask_size[0],
                                                                param.mask_size[1]);
    MSHADOW_CUDA_POST_KERNEL_CHECK(mask_class_encode);
  });
}

DMLC_REGISTER_PARAMETER(MRCNNMaskTargetParam);

NNVM_REGISTER_OP(_contrib_mrcnn_mask_target)
.describe("Generate mask targets for Mask-RCNN.")
.set_num_inputs(5)
.set_num_outputs(2)
.set_attr_parser(ParamParser<MRCNNMaskTargetParam>)
.set_attr<mxnet::FInferShape>("FInferShape", MRCNNMaskTargetShape)
.set_attr<nnvm::FInferType>("FInferType", MRCNNMaskTargetType)
.set_attr<FCompute>("FCompute<gpu>", MRCNNMaskTargetCompute<gpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("rois", "NDArray-or-Symbol", "Bounding box coordinates, a 3D array")
.add_argument("masks_meta", "NDArray-or-Symbol", "List of tuples (mask_idx, start_idx, count)")
.add_argument("masks_coords",  "NDArray-or-Symbol", "List of (x,y) coordinates")
.add_argument("matches", "NDArray-or-Symbol", "Index to a gt_mask, a 2D array")
.add_argument("cls_targets", "NDArray-or-Symbol",
              "Value [0, num_class), excluding background class, a 2D array")
.add_arguments(MRCNNMaskTargetParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
