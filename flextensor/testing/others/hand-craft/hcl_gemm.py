"""
Test heterocl
"""
import heterocl as hcl
import numpy as np
import time

def gemm(m=1024, n=1024, k=1024, dtype=hcl.Int(), target=None):
    matrix_1 = hcl.placeholder((m, k), dtype=dtype)
    matrix_2 = hcl.placeholder((k, n), dtype=dtype)

    def kernel(matrix_1, matrix_2):
       r = hcl.reduce_axis(0, k, 'k')
  
       mat1_buf =  hcl.compute((m, k), 
              lambda x, y: matrix_1[x, y],
              dtype=dtype,
              name="mat1_buf")

       mat2_buf =  hcl.compute((k, n), 
              lambda x, y: matrix_2[x, y],
              dtype=dtype,
              name="mat2_buf")

       return hcl.compute((m, n),
                lambda x, y: hcl.sum(mat1_buf[x, r] * mat2_buf[r, y],
                                     axis=r, dtype=dtype),
                dtype=dtype,
                name="out_matrix")

    s = hcl.create_schedule([matrix_1, matrix_2], kernel)
    out_matrix = kernel.out_matrix
    mat1_buf = kernel.mat1_buf
    mat2_buf = kernel.mat2_buf

    m_block = 4
    n_block = 8
    k_block = 16

    m , k = s[mat1_buf].op.axis
    #print(m , k)
    x0, x1  = s[mat1_buf].split(m, factor=m_block)
    z0, z1  = s[mat1_buf].split(k, factor=k_block)
    s[mat1_buf].reorder(x0, z0, x1, z1)

    k , n = s[mat2_buf].op.axis
    z0, z1 = s[mat2_buf].split(k, factor=k_block)
    y0, y1 = s[mat2_buf].split(n, factor=n_block)
    s[mat2_buf].reorder(y0, z0, y1, z1)


    m, n, k = s[out_matrix].op.axis
    #print(m, n, k)
    #s[out_matrix].reorder(n, m, k)

    x0, x1 = s[out_matrix].split(m, factor=m_block)
    y0, y1 = s[out_matrix].split(n, factor=n_block)
    z0, z1 = s[out_matrix].split(k, factor=k_block)

    s[out_matrix].reorder( x0, y0, z0,  x1, y1, z1)
    
    s[mat1_buf].compute_at(s[out_matrix], s[out_matrix].op.axis[0])
    

    #s[mat1_buf].compute_at(s[out_matrix], z0)

    #s[mat1_buf].compute_at(s[out_matrix], z1)

    #s[out_matrix].pipeline(x1)

    f = hcl.build(s, target=target)
    print(type(f))
    print(f)
    #code = hcl.lower(s)
    #print(code) 
    return f

def time_gemm(dtype, m=1024, n=1024, k=1024, target=None):
    hcl.init(dtype)
    f = gemm(m, n, k, dtype, target)
    np_1 = np.random.randint(10, size=(m, k))
    np_2 = np.random.randint(10, size=(k, n))
    np_3 = np.matmul(np_1, np_2)

    hcl_m1 = hcl.asarray(np_1, dtype=dtype)
    hcl_m2 = hcl.asarray(np_2, dtype=dtype)
    hcl_m3 = hcl.asarray(np.zeros((m, n)), dtype=dtype)
    f(hcl_m1, hcl_m2, hcl_m3)
    begin = time.time()
    for i in range(10):
        f(hcl_m1, hcl_m2, hcl_m3)
    end = time.time()
    print("dtype is: ", dtype)
    print("average of 10 runs takes: {} sec".format((end - begin) / 10))
    np.testing.assert_allclose(hcl_m3.asnumpy(), np_3, rtol=1e-03)

###############################################################################
# Test the algorithm with different data types
#dtypes = [hcl.Int(32), hcl.Float(), hcl.Fixed(32, 16)]
dtypes = [hcl.Float()]
for dtype in dtypes:
    time_gemm(dtype, m=256, n=512, k=1024, target="vhls")
