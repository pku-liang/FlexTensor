import tvm


INTRIN_TABLE = {}


class Intrinsic(object):
    def __init__(self, category, name, func, args, intrin, target):
        self.key = "{}_{}_{}".format(category, name, target)
        self.func = func
        self.args = args
        self.intrin = intrin
        self.target = target
        self.category = category


def register_intrin(intrin, override=False):
    if intrin.key in INTRIN_TABLE and not override:
        print("[Warning]: Same intrinsic occurs again %s" % intrin.key)
    if intrin.target not in INTRIN_TABLE:
      INTRIN_TABLE[intrin.target] = []
    INTRIN_TABLE[intrin.target].append(intrin)


def register(func, args, category, name, intrin, target, override=False):
    intrinsic = Intrinsic(category, name, func, args, intrin, target)
    register_intrin(intrinsic, override=override)


def intrinsic_gemm_int8_compute(i, j, k):
  a = tvm.placeholder((i, k), name='a', dtype="int8")
  b = tvm.placeholder((k, j), name='b', dtype="int8")
  kk = tvm.reduce_axis((0, k), name='k')
  c = tvm.compute((i, j), lambda ii, jj:
                  tvm.sum(a[ii, kk] * b[kk, jj], axis=kk), name='c')

  return c, [a, b, c]


def intrinsic_gemm_int8(i, j, k, il, jl, kl, ic, jc, kc, dim):
  """
  (i, k) * (k, j)
  i, j, k: normal iteration size
  il, jl, kl: last iteration size
  ic, jc, kc: last iteration condition
  """
  assert i * k + k * j <= 256 * 1024, 'input too large for scratchpad'
  assert 4 * (i * j) <= 64 * 1024, 'input too large for accumulator'

  DIM = dim

  _, bufs = intrinsic_gemm_int8_compute(i, j, k)
  a, b, c = bufs

  strideA = tvm.var("sA")
  Ab = tvm.decl_buffer(a.shape, a.dtype,
                        name="A",
                        offset_factor=1,
                        strides=[strideA, 1])
  strideB = tvm.var("sB")
  Bb = tvm.decl_buffer(b.shape, b.dtype,
                        name="B",
                        offset_factor=1,
                        strides=[strideB, 1])
  strideC = tvm.var("sC")
  Cb = tvm.decl_buffer(c.shape, c.dtype,
                        name="C",
                        offset_factor=1,
                        strides=[strideC, 1])

  II = i // DIM + (0 if i % DIM == 0 else 1)
  JJ = j // DIM + (0 if j % DIM == 0 else 1)
  KK = k // DIM + (0 if k % DIM == 0 else 1)
  pad_I = 0 if i % DIM == 0 else (DIM - i % DIM)
  pad_J = 0 if j % DIM == 0 else (DIM - j % DIM)
  pad_K = 0 if k % DIM == 0 else (DIM - k % DIM)

  IIl = il // DIM + (0 if il % DIM == 0 else 1)
  JJl = jl // DIM + (0 if jl % DIM == 0 else 1)
  KKl = kl // DIM + (0 if kl % DIM == 0 else 1)
  pad_Il = 0 if il % DIM == 0 else (DIM - il % DIM)
  pad_Jl = 0 if jl % DIM == 0 else (DIM - jl % DIM)
  pad_Kl = 0 if kl % DIM == 0 else (DIM - kl % DIM)

  II = tvm.if_then_else(ic, IIl, II)
  JJ = tvm.if_then_else(jc, JJl, JJ)
  KK = tvm.if_then_else(kc, KKl, KK)
  pad_I = tvm.if_then_else(ic, pad_Il, pad_I)
  pad_J = tvm.if_then_else(jc, pad_Jl, pad_J)
  pad_K = tvm.if_then_else(kc, pad_Kl, pad_K)

  # reset-update-finalize
  def intrin_func(ins, outs):
    aa, bb = ins
    cc, = outs

    def _body():
      ib = tvm.ir_builder.create()
      # int32_t matmul_kernel(const elem_t *A, const elem_t *B, const acc_t *D,
      #          elem_t *C, int32_t I, int32_t J, int32_t K, int32_t pad_I,
      #          int32_t pad_J, int32_t pad_K, int32_t A_row_len,
      #          int32_t B_row_len, int32_t D_row_len, int32_t C_row_len,
      #          bool no_bias, bool repeating_bias);
      # D is set to a dummy address 1 to determine whether to overwrite
      # accumulator contents: on the first run, 1 will be retained and
      # overwrite the value in the accumulator; on subsequent runs D will be
      # replaced by NULL and C will accumulate on top of the accumulator's contents
      # This is controlled via bit 1 << (ADDR_LEN - 2) - see kernel source
      ib.emit(tvm.call_extern("int32", "matmul_kernel",
                              aa.access_ptr("r"),
                              bb.access_ptr("r"),
                              1,
                              cc.access_ptr("rw"),
                              II, JJ, KK,
                              pad_I, pad_J, pad_K,
                              strideA, strideB, 0, strideC,
                              True, False))
      return ib.get()

    def _reset():
      ib = tvm.ir_builder.create()
      # int32_t matmul_reset(elem_t *C, int32_t I, int32_t J, int32_t pad_I,
      #         int32_t pad_J, int32_t C_row_len);
      ib.emit(tvm.call_extern("int32", "matmul_reset",
                              cc.access_ptr("w"),
                              II, JJ,
                              pad_I, pad_J,
                              strideC))
      return ib.get()

    def _finalize():
      ib = tvm.ir_builder.create()
      # Move out C from accumulator
      # int32_t matmul_finalize(elem_t *C, int32_t I, int32_t J, int32_t pad_I,
      #         int32_t pad_J, int32_t C_row_len);
      ib.emit(tvm.call_extern("int32", "matmul_finalize",
                              cc.access_ptr("rw"),
                              II, JJ,
                              pad_I, pad_J,
                              strideC))
      return ib.get()
    # standalone (without reduce axis split), reset, update
    return None, _reset(), _body(), _finalize()
  with tvm.build_config(offset_factor=1):
    return tvm.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb}, name="sp_gemm")


def generate_intrinsic_gemm_int8_dim16(N, M, K, fN, fM, fK, axisN, axisM, axisK):
  last_n = N % fN
  nc = tvm.expr.EQ(axisN, N // fN) if last_n != 0 else False
  last_n = last_n if last_n != 0 else fN

  last_l = K % fK
  lc = tvm.expr.EQ(axisK, K // fK) if last_l != 0 else False
  last_l = last_l if last_l != 0 else fK

  last_m = M % fM
  mc = tvm.expr.EQ(axisM, M // fM) if last_m != 0 else False
  last_m = last_m if last_m != 0 else fM

  gemm = intrinsic_gemm_int8(fN, fM, fK, last_n,
                        last_m, last_l, nc, mc, lc, 16)

  return gemm


register(
  intrinsic_gemm_int8_compute,
  (32, 32, 32),
  "gemmini",
  "gemm_size16",
  generate_intrinsic_gemm_int8_dim16,
  "c -device=micro_dev"
  )