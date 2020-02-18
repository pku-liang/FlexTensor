import heterocl as hcl


def zero_pad2d(inputs, padding=0):
    """Zero padding for 2d tensor

    Args:
    -----------------------------
    inputs : hcl.tensor.Tensor
        shape [batch, channel, height, width]
    padding: (optional:0) int or tuple
        expected: (h_pad_up, h_pad_down, w_pad_up, w_pad_down)
    -----------------------------

    Returns:
    -----------------------------
    hcl.tensor.Tensor
        shape [batch, channel, padded_height, padded_width]
    -----------------------------
    """
    padding = (padding, padding, padding, padding) if isinstance(padding, (int, )) else padding
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])

    padding_zero = 0.0 if "float" in inputs.dtype else 0

    batch_size, in_channel, height, width = inputs.shape
    return hcl.compute(
        (batch_size, in_channel, height + padding[0] + padding[1], width + padding[2] + padding[3]),
        lambda b, c, h, w: hcl.select(
                            hcl.tvm.all(h >= padding[0], h < height + padding[0], w >= padding[2], w < width + padding[2]),
                            inputs[b, c, h - padding[0], w - padding[2]],
                            padding_zero
                            ),
        name='Padding'
        )


def kernel_conv2d(inputs, weight, stride=1, padding=1, dilation=1, groups=1):
    """Convolution 2d NCHW layout

    Args:
    -----------------------------
    inputs  : hcl.tensor.Tensor
        shape [batch, channel, height, width]
    weight  : hcl.tensor.Tensor
        shape [out_channel, channel // groups, kernel_height, kernel_width]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int

    groups  : (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    hcl.tensor.Tensor
        shape [batch, out_channel, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_h, k_w = weight.shape
    out_channel_per_group = out_channel // groups

    stride = (stride, stride) if isinstance(stride, (int, )) else stride
    padding = (padding, padding) if isinstance(padding, (int, )) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, )) else dilation

    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1
    rc = hcl.reduce_axis(*(0, channel_per_group), name="rc")
    rh = hcl.reduce_axis(*(0, k_h), name="rh")
    rw = hcl.reduce_axis(*(0, k_w), name="rw")

    padded = zero_pad2d(inputs, padding=padding)
    output = hcl.compute(
        (batch_size, out_channel, out_h, out_w),
        lambda b, c, h, w: hcl.sum(
            (padded[b, c // out_channel_per_group * channel_per_group + rc, 
                    h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]]
            * weight[c, rc, rh, rw]),
            axis=[rc, rw, rh]
        ), "C"
    )
    return output


def main():
    N, C, H, W = 1, 1024, 7, 7
    R, S = 3, 3
    K = 1024
    A = hcl.placeholder((N, C, H, W), dtype="float32", name="A")
    W = hcl.placeholder((K, C, R, S), dtype="float32", name="W")

    s = hcl.create_schedule([A, W], kernel_conv2d)
    # split
    C = kernel_conv2d.C
    
    b, k, h, w, rc, rw, rh = s[C].op.axis
    bo, bi = s[C].split(b, factor=1)
    ko, ki = s[C].split(k, factor=32)
    ho, hi = s[C].split(h, factor=1)
    wo, wi = s[C].split(w, factor=7)
    rco, rci = s[C].split(rc, factor=32)
    rwo, rwi = s[C].split(rw, factor=3)
    rho, rhi = s[C].split(rh, factor=1)

    # reorder shuffle
    s[C].reorder(bo, ko, ho, wo, bi, ki, hi, wi)
    
    # reorder local
    s[C].reorder(bo, ho, wo, ko)
    s[C].reorder(bi, hi, wi, rco, ki, rci, rwo, rwi, rho, rhi)
    
    # reshape
    # nothing
    
    # partition
    s.partition(A, dim=3)

    # pipeline
    s[C].pipeline(ko)

    # reuse_at
    # wrong
    # WA = s.reuse_at(A, s[C], ko, "WA")

    print(hcl.build(s, target="vhls"))




if __name__ == "__main__":
    main()
    """
    // result:
    #include <ap_int.h>
    #include <ap_fixed.h>
    #include <math.h>

    void default_function(float A[1][1024][7][7], float W[1024][1024][3][3], ap_int<32> C[1][1024][7][7]) {
    #pragma HLS array_partition variable=A complete dim=3
    ap_int<32> Padding[1][1024][9][9];
    for (ap_int<32> c = 0; c < 1024; ++c) {
        for (ap_int<32> h = 0; h < 9; ++h) {
        for (ap_int<32> w = 0; w < 9; ++w) {
            Padding[0][c][h][w] = ((ap_int<32>)(((((1 <= h) && (h < 8)) && (1 <= w)) && (w < 8)) ? A[(((((w - ((w + -1) % 7)) + (h * 7)) + (c * 49)) + -8) / 50176)][((((((w - ((w + -1) % 7)) + (h * 7)) + (c * 49)) + -8) / 49) % 1024)][((((((w - ((w + -1) % 7)) + (h * 7)) + (c * 49)) + -8) / 7) % 7)][((w + -1) % 7)] : 0.000000e+00f));
        }
        }
    }
    for (ap_int<32> h_outer = 0; h_outer < 7; ++h_outer) {
        for (ap_int<32> c_outer = 0; c_outer < 32; ++c_outer) {
        #pragma HLS pipeline
        for (ap_int<32> w_inner = 0; w_inner < 7; ++w_inner) {
            for (ap_int<32> rc_outer = 0; rc_outer < 32; ++rc_outer) {
            ap_int<32> sum;
            sum = 0;
            for (ap_int<32> c_inner = 0; c_inner < 32; ++c_inner) {
                for (ap_int<32> rc_inner = 0; rc_inner < 32; ++rc_inner) {
                for (ap_int<32> rw_inner = 0; rw_inner < 3; ++rw_inner) {
                    for (ap_int<32> rh_outer = 0; rh_outer < 3; ++rh_outer) {
                    sum = ((ap_int<32>)((((float)Padding[0][(rc_inner + (rc_outer * 32))][(h_outer + rh_outer)][(w_inner + rw_inner)]) * W[(c_inner + (c_outer * 32))][(rc_inner + (rc_outer * 32))][rh_outer][rw_inner]) + ((float)sum)));
                    }
                }
                }
            }
            C[0][(c_inner + (c_outer * 32))][h_outer][w_inner] = sum;
            }
        }
        }
    }
    }

    """
