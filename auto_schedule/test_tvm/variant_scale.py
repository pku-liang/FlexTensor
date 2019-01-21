# import tvm
import os
import numpy as np
import matplotlib.pyplot as plt
# from auto_schedule.training_examples import FUNC_TABLE


class Point(object):
    def __init__(self, factor_lst, performance):
        self.xs = tuple(factor_lst)
        self.y = performance
        self.dim = len(factor_lst)

    def __str__(self):
        ret = "[xs=" + str(self.xs) + "; y=" + str(self.y) + "]"
        return ret

    @classmethod
    def from_str(cls, s):
        lst = s[1:-1]
        lst = lst.split("; ")
        xs = lst[0].split("=")[1]
        xs = xs[1:-1]
        xs = xs.split(", ")
        tmp = []
        for x in xs:
            tmp.append(int(x))
        p = lst[1].split("=")[1]
        p = float(p)
        return cls(tmp, p)


class Curve(object):
    def __init__(self, name, shape, point_lst):
        self.name = name
        self.shape = tuple(shape)
        self.point_lst = point_lst

    def __str__(self):
        ret = "[name=" + self.name + "; shape=" + str(self.shape)
        for point in self.point_lst:
            ret += "; " + str(point)
        ret += "]"
        return ret

    @classmethod
    def from_str(cls, s):
        lst = s[1:-1]
        lst = lst.split("; ")
        name = lst[0].split("=")[1]
        shape = lst[1].split("=")[1]
        shape = shape[1:-1]
        shape = shape.split(", ")
        tmp = []
        for e in shape:
            tmp.append(int(e))
        points = []
        length = (len(lst) - 2) // 2
        for i in range(length):
            p = lst[2 * i + 2] + "; " + lst[2 * i + 3]
            point = Point.from_str(p)
            points.append(point)
        return cls(name, tmp, points)


def test_matmul(N_end, M_end, L_end):
    N_end, M_end, L_end = max(N_end, 16), max(M_end, 16), max(L_end, 16)
    N_beg, M_beg, L_beg = 512, 512, 512
    fn_beg, fm_beg, fl_beg = 1, 1, 1
    ascend_factor = 2
    prime_add_lst = [0, 1, 5, 9]
    matmul = FUNC_TABLE["matmul"].func
    curve_lst = []
    N_power_lst = list(range(int(np.log(N_end / N_beg) / np.log(ascend_factor)) + 1))
    M_power_lst = list(range(int(np.log(M_end / M_beg) / np.log(ascend_factor)) + 1))
    L_power_lst = list(range(int(np.log(L_end / L_beg) / np.log(ascend_factor)) + 1))
    for N_power in N_power_lst:
        M_power = L_power = N_power
        N_ = N_beg * np.power(ascend_factor, N_power)
        M_ = M_beg * np.power(ascend_factor, M_power)
        L_ = L_beg * np.power(ascend_factor, L_power)
        for prime_add in prime_add_lst:
            N = N_ + prime_add
            M = M_ + prime_add
            L = L_ + prime_add
            point_lst = []
            fn_end, fm_end, fl_end = 128, 128, 128
            fn_inc, fm_inc, fl_inc = 2, 2, 2
            length = int(np.log(fn_end / fn_beg) / np.log(fn_inc)) + 2
            for i in range(length):
                j = k = i
                fn = np.power(fn_inc, i) * fn_beg
                fm = np.power(fm_inc, j) * fm_beg
                fl = np.power(fl_inc, k) * fl_beg
                ops, arg_bufs = matmul(N, M, L)
                s = tvm.create_schedule(ops)
                a, b = s[ops].op.axis
                c = s[ops].op.reduce_axis[0]
                ao, ai = s[ops].split(a, factor=fn)
                bo, bi = s[ops].split(b, factor=fm)
                co, ci = s[ops].split(c, factor=fl)
                s[ops].reorder(ao, bo, co, ai, bi, ci)
                func = tvm.build(s, arg_bufs, "llvm")
                ctx = tvm.context("llvm", 0)
                evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
                A = np.random.uniform(-10, 10, size=(N, M)).astype("float32")
                B = np.random.uniform(-10, 10, size=(M, L)).astype("float32")
                C = np.zeros(shape=(N, L)).astype("float32")
                tvm_A = tvm.nd.array(A, ctx)
                tvm_B = tvm.nd.array(B, ctx)
                tvm_C = tvm.nd.array(C, ctx)
                time_cost = evaluator(tvm_A, tvm_B, tvm_C).mean * 1e3
                point = Point((fn, fm, fl), time_cost)
                point_lst.append(point)
            curve = Curve("matmul", (N, M, L), point_lst)
            curve_lst.append(curve)
    return curve_lst


def save_curves(curve_lst, filename):
    with open(filename, "w") as f:
        for curve in curve_lst:
            s = str(curve) + "\n"
            f.write(s)


def load_curve(filename):
    curve_lst = []
    with open(filename, "r") as f:
        for line in f.readlines():
            curve = Curve.from_str(line.strip())
            curve_lst.append(curve)
    return curve_lst


def draw_curve(curve, dirname):
    dim = len(curve.point_lst[0].xs)
    value_lst = [set() for i in range(dim)]
    for point in curve.point_lst:
        for i in range(dim):
            value_lst[i].add(point.xs[i])
    lst = []
    for i in range(dim):
        lst.append(list(value_lst[i]))
    shape = []
    for e in lst:
        shape.append(max(e) + 1)
    f = np.zeros(shape=shape)
    for point in curve.point_lst:
        f[point.xs] = point.y
    lines = np.reshape(f, (-1, shape[-1]))
    plt.figure()
    for j, line in enumerate(lines):
        xs = []
        ys = []
        for i, y in enumerate(line):
            if y != 0:
                xs.append(i)
                ys.append(y)
        if not xs:
            continue
        plt.plot(xs, ys)
        v = j
        s = ""
        for e in reversed(shape[:-1]):
            tmp = v % e
            v //= e
            s += str(tmp)
        s = str(list(reversed(s)))
        plt.savefig(os.path.join(dirname, "line_{}_{}.jpg".format(j, s)))
    plt.close("all")


def draw_curve_block(curve, dirname):
    block_lst = []
    for point in curve.point_lst:
        block_lst.append((point.xs[0], point.y))
    value_lst = list(sorted(block_lst, key=lambda x: x[0]))
    xs = []
    ys = []
    for (x, y) in value_lst:
        xs.append(x)
        ys.append(y)

    plt.figure()
    plt.plot(xs, ys)
    plt.savefig(os.path.join(dirname, "{}.jpg".format(dirname)))
    plt.close("all")


def main():
    # curve_lst = test_matmul(2**12, 2**12, 2**12)
    filename = "matmul_curves_square.txt"
    # save_curves(curve_lst, filename)
    load_lst = load_curve(filename)
    for curve in load_lst:
        dirname = curve.name + "_" + str(curve.shape)
        os.mkdir(dirname)
        draw_curve_block(curve, dirname)


if __name__ == "__main__":
    main()



