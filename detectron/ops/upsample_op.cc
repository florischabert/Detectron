
/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "upsample.h"

namespace caffe2 {

template <>
bool UpsampleOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  auto* Y = Output(0);

  const int batch_size = X.dim32(0),
            num_channels = X.dim32(1),
            input_height = X.dim32(2),
            input_width = X.dim32(3);
  int output_width = input_width * width_scale_;
  int output_height = input_height * height_scale_;
  Y->Resize(batch_size, num_channels, output_height, output_width);

  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < num_channels; ++c) {
      for (int y = 0; y < output_height; ++y) {
        const int in_y = std::min((int)(y / height_scale_), (input_height - 1));
        for (int x = 0; x < output_width; ++x) {
          const int in_x = std::min((int)(x / width_scale_), (input_width - 1));
          Ydata[output_width * y + x] = Xdata[input_width * in_y + in_x];
        }
      }
      Xdata += input_height * input_width;
      Ydata += output_width * output_height;
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(Upsample, UpsampleOp<float, CPUContext>);

OPERATOR_SCHEMA(Upsample)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("width_scale", "Scale along width dimension")
    .Arg("height_scale", "Scale along height dimension")
    .Input(
        0,
        "X",
        "1D input tensor")
    .Output(
        0,
        "Y",
        "1D output tensor");

} 