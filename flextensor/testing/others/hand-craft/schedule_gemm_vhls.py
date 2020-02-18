import heterocl as hcl


def kernel_gemm(A, B):
    k = hcl.reduce_axis(0, A.shape[1], "k")
    return hcl.compute(
        (A.shape[0], B.shape[1]),
        lambda i, j: hcl.sum(A[i, k] * B[k, j], axis=k),
        "C")


def main():
    M = 512
    N = 512
    K = 512
    A = hcl.placeholder((M, K), dtype="float32", name="A")
    B = hcl.placeholder((K, N), dtype="float32", name="B")

    s = hcl.create_schedule([A, B], kernel_gemm)
    # split
    C = kernel_gemm.C
    m, n, k = s[C].op.axis
    mo, mi = s[C].split(m, factor=16)
    no, ni = s[C].split(n, factor=32)
    ko, ki = s[C].split(k, factor=8)
    

    # reorder shuffle
    s[C].reorder(mo, no, mi, ni, ko, ki)
    

    # reorder local
    s[C].reorder(mi, ko, ki, ni)
    

    # reshape
    s.reshape(C, [512//16, 16, 512//32, 32])
    

    # partition
    s.partition(A, dim=3)
    

    # pipeline
    s[C].pipeline(mi)
    

    # reuse_at
    # nothing to do

    print(hcl.build(s, target="vhls"))




if __name__ == "__main__":
    main()
    """
    // result:
    #include <ap_int.h>
    #include <ap_fixed.h>
    #include <math.h>

    void default_function(float A[512][512], float B[512][512], ap_int<32> C[32][16][16][32]) {
    #pragma HLS array_partition variable=A complete dim=3
    for (ap_int<32> i_outer = 0; i_outer < 32; ++i_outer) {
        for (ap_int<32> j_outer = 0; j_outer < 16; ++j_outer) {
        for (ap_int<32> i_inner = 0; i_inner < 16; ++i_inner) {
        #pragma HLS pipeline
            for (ap_int<32> k_outer = 0; k_outer < 64; ++k_outer) {
            ap_int<32> sum;
            sum = 0;
            for (ap_int<32> k_inner = 0; k_inner < 8; ++k_inner) {
                for (ap_int<32> j_inner = 0; j_inner < 32; ++j_inner) {
                sum = ((ap_int<32>)((A[(i_inner + (i_outer * 16))][(k_inner + (k_outer * 8))] * B[(k_inner + (k_outer * 8))][(j_inner + (j_outer * 32))]) + ((float)sum)));
                }
            }
            C[i_outer][i_inner][j_outer][j_inner] = sum;
            }
        }
        }
    }

    """