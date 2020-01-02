/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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
*******************************************************************************/

#include <assert.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "mkldnn.hpp"

using namespace mkldnn;

using namespace std;

memory::dim product(const memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
            std::multiplies<memory::dim>());
}

/* to use MKL-DNN the command is:
 export LD_LIBRARY_PATH=/home/CORP.PKUSC.ORG/zchno/mkl-dnn-install/lib64:$LD_LIBRARY_PATH && 
 g++ -std=c++11 -I /home/CORP.PKUSC.ORG/zchno/mkl-dnn-install/include -L /home/CORP.PKUSC.ORG/zchno/mkl-dnn-install/lib64 -lmkldnn simple_conv.cpp -o bin/simple_conv &&
 bin/simple_conv
*/

void simple_conv(const memory::dim N, const memory::dim H, const memory::dim W, const memory::dim CI,
    const memory::dim CO, const memory::dim k, const memory::dim S, const memory::dim P, int times = 100) {
    using tag = memory::format_tag;
    using dt = memory::data_type;

    engine eng(engine::cpu, 0);
    stream s(eng);

    /* Create a vector primitive to hold the network. For efficiency purpose,
     * weights are stored in a separate net to perform reordering only once. */
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    /* GoogleNet: conv24
     * {batch, 1024, 7, 7} (x) {1024, 1024, 3, 3} -> {batch, 1024, 7, 7}
     * strides: {1, 1}
     */
    //H = 7; W = 7; CI = 1024; CO = 1024; k = 3; S = 1; P = 1;
    memory::dims conv1_src_tz = { N, CI, H, W };
    memory::dims conv1_weights_tz = { CO, CI, k, k };
    memory::dims conv1_bias_tz = { CO };
    memory::dims conv1_dst_tz = { N, CO, (H - k + 2 * P) / S + 1, (W - k + 2 * P) / S + 1};
    memory::dims conv1_strides = { S, S };
    memory::dims conv1_padding = { P, P };

    /* Allocate input and output buffers for user data */
    std::vector<float> user_src(N * CI * H * W);

    /* Allocate and fill buffers for weights and bias */
    std::vector<float> conv1_weights(product(conv1_weights_tz));
    std::vector<float> conv1_bias(product(conv1_bias_tz));

    /* create memory for user data */
    auto user_src_memory = memory(
            { { conv1_src_tz }, dt::f32, tag::nchw }, eng, user_src.data());
    auto user_weights_memory
            = memory({ { conv1_weights_tz }, dt::f32, tag::oihw }, eng,
                    conv1_weights.data());
    auto conv1_user_bias_memory = memory(
            { { conv1_bias_tz }, dt::f32, tag::x }, eng, conv1_bias.data());

    /* create memory descriptors for convolution data w/ no specified format
     */
    auto conv1_src_md = memory::desc({ conv1_src_tz }, dt::f32, tag::any);
    auto conv1_bias_md = memory::desc({ conv1_bias_tz }, dt::f32, tag::any);
    auto conv1_weights_md
            = memory::desc({ conv1_weights_tz }, dt::f32, tag::any);
    auto conv1_dst_md = memory::desc({ conv1_dst_tz }, dt::f32, tag::any);

    /* create a convolution */
    auto conv1_desc = convolution_forward::desc(prop_kind::forward_inference,
            convolution_direct, conv1_src_md, conv1_weights_md, conv1_bias_md,
            conv1_dst_md, conv1_strides, conv1_padding, conv1_padding,
            padding_kind::zero);
    auto conv1_prim_desc = convolution_forward::primitive_desc(conv1_desc, eng);

    /* create reorders for data and weights if layout requested by
     * convolution is different from NCHW/OIHW */
    auto conv1_src_memory = user_src_memory;
    if (conv1_prim_desc.src_desc() != user_src_memory.get_desc()) {
        conv1_src_memory = memory(conv1_prim_desc.src_desc(), eng);
        net.push_back(reorder(user_src_memory, conv1_src_memory));
        net_args.push_back({ { MKLDNN_ARG_FROM, user_src_memory },
                { MKLDNN_ARG_TO, conv1_src_memory } });
    }

    auto conv1_weights_memory = user_weights_memory;
    if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
        reorder(user_weights_memory, conv1_weights_memory)
                .execute(s, user_weights_memory, conv1_weights_memory);
    }

    auto conv1_dst_memory = memory(conv1_prim_desc.dst_desc(), eng);

    /* create convolution primitive and add it to net */
    net.push_back(convolution_forward(conv1_prim_desc));
    net_args.push_back({ { MKLDNN_ARG_SRC, conv1_src_memory },
            { MKLDNN_ARG_WEIGHTS, conv1_weights_memory },
            { MKLDNN_ARG_BIAS, conv1_user_bias_memory },
            { MKLDNN_ARG_DST, conv1_dst_memory } });


    for (int j = 0; j < times; ++j) {
        assert(net.size() == net_args.size() && "something is missing");
        for (size_t i = 0; i < net.size(); ++i)
            net.at(i).execute(s, net_args.at(i));
    }
}

int main(int argc, char **argv) {
    int arg_lst[][7] = {
        {448, 448, 3, 64, 7, 2, 3},
        {112, 112, 64, 192, 3, 1, 1},
        {56, 56, 192, 128, 1, 1, 0},
        {56, 56, 128, 256, 3, 1, 1},
        {28, 28, 512, 256, 1, 1, 0},
        {28, 28, 256, 512, 3, 1, 1},
        {14, 14, 1024, 512, 1, 1, 0},
        {7, 7, 1024, 1024, 3, 1, 1},
    };
    for(int i = 0; i < 8; ++i)
    {
        const memory::dim N = 1;
        try {
            int times = 10;
            // warm-up
            simple_conv(N, arg_lst[i][0], arg_lst[i][1], arg_lst[i][2], arg_lst[i][3], arg_lst[i][4], arg_lst[i][5], arg_lst[i][6], 1);
            auto begin = chrono::duration_cast<chrono::milliseconds>(
                    chrono::steady_clock::now().time_since_epoch())
                                 .count();
            simple_conv(N, arg_lst[i][0], arg_lst[i][1], arg_lst[i][2], arg_lst[i][3], arg_lst[i][4], arg_lst[i][5], arg_lst[i][6], times);
            auto end = chrono::duration_cast<chrono::milliseconds>(
                    chrono::steady_clock::now().time_since_epoch())
                               .count();
            cout << "C" << i + 1 << " Use time " << (end - begin) / (times + 0.0) << "\n";
        } catch (error &e) {
            std::cerr << "status: " << e.status << std::endl;
            std::cerr << "message: " << e.message << std::endl;
        }
    }
    return 0;
}
