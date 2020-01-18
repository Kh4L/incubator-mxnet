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
 * \file rpn_target-inl.h
 * \brief RPN target generator
 * \author Serge Panev
 */

#ifndef MXNET_OPERATOR_CONTRIB_RPN_TARGET_INL_H_
#define MXNET_OPERATOR_CONTRIB_RPN_TARGET_INL_H_

#include <mxnet/operator.h>
#include <vector>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../tensor/init_op.h"
#include "./bounding_box-common.h"

namespace mxnet {
namespace op {


namespace rpn_target_index {
    enum RPNTargetOpInputs {kBbox, kAnchor};
    enum RPNTargetOpOutputs {kClsTarget, kBoxTarget, kBoxMask};
    enum RPNTargetOpResource {kTempSpace, kRandom};
}  // namespace rpn_target_index

struct RPNTargetParam : public dmlc::Parameter<RPNTargetParam> {
  mxnet::TShape image_size;
  int num_sample;
  float pos_iou_thresh;
  float neg_iou_thresh;
  float pos_ratio;
  // TODO(spanev): check if stds are really needed

  DMLC_DECLARE_PARAMETER(RPNTargetParam) {
    DMLC_DECLARE_FIELD(image_size)
    .set_expect_ndim(2).enforce_nonzero()
    .describe("Image size height and width: (w, h).");
    DMLC_DECLARE_FIELD(num_sample)
    .describe("Number of samples for RPN targets.");
    DMLC_DECLARE_FIELD(pos_iou_thresh).set_range(0.0f, 1.0f).set_default(0.7f)
    .describe("Anchor with IoU larger than `pos_iou_thresh` is regarded as a positive sample.");
    DMLC_DECLARE_FIELD(neg_iou_thresh).set_range(0.0f, 1.0f).set_default(0.3f)
    .describe("Anchor with IoU smaller than ``neg_iou_thresh` is regarded as a negative sample."
              "Anchor with IoU in between ``pos_iou_thresh` and ``neg_iou_thresh`` is ignored.");
    DMLC_DECLARE_FIELD(pos_ratio).set_range(0.0f, 1.0f).set_default(0.5f)
    .describe("`pos_ratio` defines how many positive samples (`pos_ratio * num_sample`) are "
              "to be sampled.");
  }
};

inline bool RPNTargetShape(const NodeAttrs& attrs,
                           std::vector<mxnet::TShape>* in_shapes,
                           std::vector<mxnet::TShape>* out_shapes) {
  using namespace mshadow;
  CHECK_EQ(in_shapes->size(), 2U);
  CHECK_EQ(out_shapes->size(), 3U);

  mxnet::TShape& bbox_shape = (*in_shapes)[0];
  mxnet::TShape& anchor_shape = (*in_shapes)[1];

  CHECK_GE(bbox_shape.ndim(), 2)
    << "bbox must have ndim == 2 "
    << bbox_shape.ndim() << " provided";
  int last_dim = bbox_shape[bbox_shape.ndim() - 1];
  CHECK_EQ(last_dim, 4)
    << "last dimension of bbox must be 4 "
    << last_dim << " provided";

  CHECK_GE(anchor_shape.ndim(), 2)
    << "anchor must have dim === 2 "
    << anchor_shape.ndim() << " provided";
  last_dim = anchor_shape[anchor_shape.ndim() - 1];
  CHECK_EQ(last_dim, 4)
    << "last dimension of anchor must be 4 "
    << last_dim << " provided";

  const int num_anc = anchor_shape[0];

  SHAPE_ASSIGN_CHECK(*out_shapes, 0,  mxnet::TShape(Shape1(num_anc)));
  SHAPE_ASSIGN_CHECK(*out_shapes, 1,  mxnet::TShape(Shape2(num_anc, 4)));
  SHAPE_ASSIGN_CHECK(*out_shapes, 2,  mxnet::TShape(Shape2(num_anc, 4)));
  return true;
}

inline bool RPNTargetType(const NodeAttrs& attrs,
                          std::vector<int>* in_type,
                          std::vector<int>* out_type) {
  CHECK_EQ(in_type->size(), 2);
  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "Input must have specified type";
  CHECK_EQ((*in_type)[0], (*in_type)[1]);

  out_type->clear();
  out_type->push_back(dtype);
  out_type->push_back(dtype);
  out_type->push_back(dtype);
  return true;
}

// Taken from bounding_box-inl.h compute_overlap
struct compute_iou_and_mask_invalid {
  template<typename DType>
  MSHADOW_XINLINE static bool InvalidAnchor(const DType *box, DType width, DType height) {
    DType a_xmin = box[0];
    DType a_ymin = box[1];
    DType a_xmax = box[2];
    DType a_ymax = box[3];
    return (a_xmin < 0) || (a_ymin < 0) || (a_xmax >= width) || (a_ymax >= height);
  }

  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *lhs,
                                  const DType *rhs, int num,
                                  int begin, int stride, DType width, DType height) {
    int l = i / num;
    int r = i % num;
    int l_index = l * stride + begin;
    int r_index = r * stride + begin;
    if (InvalidAnchor(lhs + l_index, width, height)) {
      out[i] = DType(-1);
      return;
    }
    int encode = box_common_enum::kCorner;
    DType intersect = Intersect(lhs + l_index, rhs + r_index, encode);
    intersect *= Intersect(lhs + l_index + 1, rhs + r_index + 1, encode);
    if (intersect <= 0) {
      out[i] = DType(0);
      return;
    }
    DType l_area = BoxArea(lhs + l_index, encode);
    DType r_area = BoxArea(rhs + r_index, encode);
    out[i] = intersect / (l_area + r_area - intersect);
  }
};

template <typename T>
inline size_t align(size_t value, int alignement) {
  value = value * sizeof(T);
  return (value + (alignement - 1)) & ~(alignement - 1);
}

template <typename xpu>
void RPNTargetRun(const RPNTargetParam& params,
                  const std::vector<TBlob> &inputs,
                  const std::vector<TBlob> &outputs,
                  const OpContext &ctx,
                  mshadow::Stream<xpu> *s);

template<typename xpu>
void RPNTargetCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext &ctx,
                            const std::vector<TBlob> &inputs,
                            const std::vector<OpReqType> &req,
                            const std::vector<TBlob> &outputs) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const RPNTargetParam& p = dmlc::get<RPNTargetParam>(attrs.parsed);
  RPNTargetRun(p, inputs, outputs, ctx, s);
}
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_RPN_TARGET_INL_H_
