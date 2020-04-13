import tvm

M = 10
N = 3
K = 7

dtype = "float32"

A = tvm.te.placeholder([M, K], dtype=dtype, name="A")
B = tvm.te.placeholder([K, N], dtype=dtype, name="B")
k = tvm.te.reduce_axis([0, K], name="k")
C = tvm.te.compute([M, N], lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=[k]), name="C")
D = tvm.te.placeholder([M, N], dtype=dtype, name="D")
E = tvm.te.compute([M, N], lambda i, j: C[i, j] + D[i, j], name="E")

dA, = tvm.te.mygradient(E, [A])

s = tvm.te.create_schedule(dA.op)

ir = tvm.lower(s, [A, B, C, D, E, dA], simple_mode=True)

print(ir)

sch = tvm.te.create_schedule(C.op)


"""
// attr [identity] storage_scope = "global"
allocate identity[float32 * 900]
// attr [C.jacobian] storage_scope = "global"
allocate C.jacobian[float32 * 2100]
// attr [C.A.grad] storage_scope = "global"
allocate C.A.grad[float32 * 2100]
produce identity {
  for (ax0, 0, 10) {
    for (ax1, 0, 3) {
      for (ax2, 0, 10) {
        for (ax3, 0, 3) {
          identity[((((ax0*90) + (ax1*30)) + (ax2*3)) + ax3)] = float32(((ax0 == ax2) && (ax1 == ax3)))
        }
      }
    }
  }
}
produce C.jacobian {
  for (i, 0, 10) {
    for (j, 0, 3) {
      for (jac_i0, 0, 10) {
        for (jac_i1, 0, 7) {
          C.jacobian[((((i*210) + (j*70)) + (jac_i0*7)) + jac_i1)] = 0f
          for (k, 0, 7) {
            C.jacobian[((((i*210) + (j*70)) + (jac_i0*7)) + jac_i1)] = (C.jacobian[((((i*210) + (j*70)) + (jac_i0*7)) + jac_i1)] + (float32(((jac_i0 == i) && (jac_i1 == k)))*B[((k*3) + j)]))
          }
        }
      }
    }
  }
}
produce C.A.grad {
  for (ax0, 0, 10) {
    for (ax1, 0, 3) {
      for (ax2, 0, 10) {
        for (ax3, 0, 7) {
          C.A.grad[((((ax0*210) + (ax1*70)) + (ax2*7)) + ax3)] = 0f
          for (k0, 0, 10) {
            for (k1, 0, 3) {
              C.A.grad[((((ax0*210) + (ax1*70)) + (ax2*7)) + ax3)] = (C.A.grad[((((ax0*210) + (ax1*70)) + (ax2*7)) + ax3)] + (identity[((((ax0*90) + (ax1*30)) + (k0*3)) + k1)]*C.jacobian[((((k0*210) + (k1*70)) + (ax2*7)) + ax3)]))
            }
          }
        }
      }
    }
  }
}
produce C.jacobian {
  for (i, 0, 10) {
    for (j, 0, 3) {
      for (jac_i0, 0, 7) {
        for (jac_i1, 0, 3) {
          C.A.grad[((((i*63) + (j*21)) + (jac_i0*3)) + jac_i1)] = 0f
          for (k, 0, 7) {
            C.A.grad[((((i*63) + (j*21)) + (jac_i0*3)) + jac_i1)] = (C.A.grad[((((i*63) + (j*21)) + (jac_i0*3)) + jac_i1)] + (A[((i*7) + k)]*float32(((jac_i0 == k) && (jac_i1 == j)))))
          }
        }
      }
    }
  }
}
produce C.B.grad {
  for (ax0, 0, 10) {
    for (ax1, 0, 3) {
      for (ax2, 0, 7) {
        for (ax3, 0, 3) {
          C.jacobian[((((ax0*63) + (ax1*21)) + (ax2*3)) + ax3)] = 0f
          for (k0, 0, 10) {
            for (k1, 0, 3) {
              C.jacobian[((((ax0*63) + (ax1*21)) + (ax2*3)) + ax3)] = (C.jacobian[((((ax0*63) + (ax1*21)) + (ax2*3)) + ax3)] + (identity[((((ax0*90) + (ax1*30)) + (k0*3)) + k1)]*C.A.grad[((((k0*63) + (k1*21)) + (ax2*3)) + ax3)]))
            }
          }
        }
      }
    }
  }
}

"""