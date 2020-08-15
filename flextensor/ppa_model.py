# from ctypes import *
import csv
import numpy as np
import os
import subprocess
import tempfile
import random


class PPAModel:

    def __init__(self, lib_path, hw_file):
        # self.model = cdll.LoadLibrary(lib_path)
        self.func = lib_path
        # self.func.restype = c_int
        self.hw_file = hw_file
        # print("Model is ready.")

    def generate_dsfl(self, info, accelerator, directives):

        if accelerator["dataflow"] not in ["gemmini_ws"]:
            print("Unsupported accelerator dataflow.")
            return

        if accelerator["dataflow"] == "gemmini_ws":
            # how many dimensions /  what's the mapped three dimesions / dimesion sizes

            # uninvolved_dim = set()
            # for idx in info['outer']:
            #      uninvolved_dim.add(idx["origin"])

            # uninvolved_dim = {'k', 'rc', 'rr', 'rs', 'q', 'p'}
            # idx_map = {'k': 'K', 'rc': 'C', 'rr': 'R', 'rs': 'S', 'q': 'Y', 'p': 'X'}

            uninvolved_dim = {'k', 'rc', 'rh', 'rw', 'h', 'w'}
            idx_map = {'k': 'K', 'rc': 'C', 'rh': 'R',
                       'rw': 'S', 'h': 'Y', 'w': 'X'}

            for idx in info['inner']:
                uninvolved_dim.discard(idx["origin"])

            directives.append("SpatialMap(1, 1) " +
                              idx_map[uninvolved_dim.pop()] + ";\n")

            for dim in uninvolved_dim:
                directives.append("TemporalMap(1, 1) " + idx_map[dim] + ";\n")

            for idx in info['inner']:
                directives.append("TemporalMap(" + str(idx["length"]) + ", " + str(
                    idx["length"]) + ") " + idx_map[idx["origin"]] + ";\n")

            directives.append("Cluster("+accelerator["PEs"]+", L);\n")
            directives.append(
                "SpatialMap(" + accelerator["x"] + ", " + accelerator["x"] + ") " + idx_map["w"] + ";\n")
            directives.append(
                "SpatialMap(" + accelerator["y"] + ", " + accelerator["y"] + ") " + idx_map["k"] + ";\n")

    def parse_info(self, info):
        name = "TEST"
        dims = {'k': 1, 'rc': 1, 'rh': 1, 'rw': 1, 'h': 1, 'w': 1}
        for idx in info['outer']:
            if idx["length"] != 1:
                dims[idx["origin"]] *= idx["length"]
        for idx in info['inner']:
            dims[idx["origin"]] *= idx["length"]
        return name, str(dims['k']), str(dims['rc']), str(dims['rh']), str(dims['rw']), str(dims['h']), str(dims['w'])

    def generate_single_mapping(self, info, accelerator, directives):
        # generator : GEMMINI_OS   GEMMINI_WS

        # if compute not in ["conv", "fc", "matmul"] :
        #     print("Tensor computation " + compute + "is not supported yet.")
        #     return

        compute_name, k, c, r, s, y, x = self.parse_info(info)

        directives.append("Layer " + compute_name + " {\n")
        directives.append("Type: CONV \n")
        directives.append("Dimensions { K " + k + ", C " + c +
                          ", R " + r + ", S " + s + ", Y " + y + ", X " + x + " }\n")
        directives.append("Dataflow {\n")

        self.generate_dsfl(info, accelerator, directives)

        directives.append("}\n")
        directives.append("}\n")

    def analyze(self, bench_name, mapping_file, infos, accelerator):

        mapping_fp = open(mapping_file, "w+")
        mapping_fp.write("Network " + bench_name + " {\n")

        # generate mappings
        for info in infos:
            directives = []
            self.generate_single_mapping(info, accelerator, directives)
            for line in directives:
                mapping_fp.write(line)

        mapping_fp.write("}\n")
        mapping_fp.close()

        # print("Start modeling.")
        param_list = [
            self.func,
            "--print_res=false",
            "--print_res_csv_file=true",
            "--print_log_file=false",
            "--Mapping_file=" + mapping_file,
            "--HW_file=" + self.hw_file,
            # "--noc_bw=32",
            "--noc_hops=1",
            "--noc_hop_latency=1",
            # "--l1_size=1024",
            # "--l2_size=1024",
            # "--num_pes=256",
            "--print_design_space=true",
            "--msg_print_lv=0"
        ]

        # print(param_list)
        # param_type = (c_char_p * 13)
        # print(param_type)
        # argc = c_int(13)
        # argv = param_type()
        # self.func.argtypes = [c_int , param_type]
        # for key, item in enumerate(param_list):
        #     argv[key] =item.encode('utf-8')

        p = subprocess.Popen(''.join(p+' ' for p in param_list), shell=True)
        p.wait()

        result_csv = mapping_file.split('/')[-1].replace(".m", ".csv")
        if not os.path.exists(result_csv):
            return None, None, None, None, None

        latency = []
        throughput = []
        power = []
        energy = []
        area = []

        with open(result_csv, "r") as f:
            reader = list(csv.reader(f))
            for row in reader[1:]:
                runtime = int(row[3])
                latency.append(runtime)
                energy.append(float(row[4]))
                throughput.append(float(row[5]) * runtime)
                area.append(float(row[7]))
                power.append(float(row[8]) * runtime)

        os.remove(result_csv)

        total_runtime = np.sum(latency)
        return total_runtime, np.mean(throughput) / total_runtime, np.mean(power) / total_runtime, np.sum(energy), np.max(area)


MAESTRO_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "./maestro")
CONFIG_FILE = "micro-1.m"


def set_config_file(file):
    global CONFIG_FILE
    CONFIG_FILE = file


def measure_latency(info):
    if info is None:
        return None
    model = PPAModel(os.path.join(MAESTRO_PATH, "./maestro"),
                     os.path.join(MAESTRO_PATH, f"./data/hw/{CONFIG_FILE}"))
    accelerator = {"dataflow": "gemmini_ws",
                   "PEs": "256", "x": "16", "y": "16"}
    no = int(random.uniform(0, 10000000))
    path = f"/tmp/test_gemmini_ws_{no}.m"
    l, *_ = model.analyze(
        "test", path, [info], accelerator)
    # os.remove(path)
    return l


if __name__ == '__main__':

    model = PPAModel("./maestro", "./data/hw/accelerator_1.m")

    # infos = [{"outer": [{"iter_var": "k.outer", "origin": "k", "brother": "k.inner", "length": 4},
    #                     {"iter_var": "p.inner", "origin": "p", "brother": "p.outer", "length": 28},
    #                     {"iter_var": "rc.outer", "origin": "rc", "brother": "rc.inner", "length": 5},
    #                     {"iter_var": "rr.inner", "origin": "rr", "brother": "rr.outer", "length": 3},
    #                     {"iter_var": "rs.inner", "origin": "rs", "brother": "rs.outer", "length": 3}],
    #          "inner": [{"iter_var": "q.inner", "origin": "q", "brother": "q.outer", "length": 32},
    #                    {"iter_var": "k.inner", "origin": "k", "brother": "k.outer", "length": 32},
    #                    {"iter_var": "rc.inner", "origin": "rc", "brother": "rc.outer", "length": 28}]}
    #         ]

    infos = [{"outer": [{"iter_var": "n.outer", "origin": "n", "brother": "n.inner", "length": 1},
                        {"iter_var": "h.outer", "origin": "h",
                            "brother": "h.inner", "length": 7},
                        {"iter_var": "w.outer", "origin": "w",
                            "brother": "w.inner", "length": 4},
                        {"iter_var": "k.outer", "origin": "k",
                            "brother": "k.inner", "length": 4},
                        {"iter_var": "n.inner", "origin": "n",
                            "brother": "n.outer", "length": 1},
                        {"iter_var": "h.inner", "origin": "h",
                            "brother": "h.outer", "length": 4},
                        {"iter_var": "rh.outer", "origin": "rh",
                         "brother": "rh.inner", "length": 1},
                        {"iter_var": "rw.outer", "origin": "rw",
                         "brother": "rw.inner", "length": 3},
                        {"iter_var": "rc.outer", "origin": "rc",
                         "brother": "rc.inner", "length": 16},
                        {"iter_var": "rh.inner", "origin": "rh",
                         "brother": "rh.outer", "length": 3},
                        {"iter_var": "rw.inner", "origin": "rw", "brother": "rw.outer", "length": 1}],
              "inner": [{"iter_var": "w.inner", "origin": "w", "brother": "w.outer", "length": 7},
                        {"iter_var": "k.inner", "origin": "k",
                         "brother": "k.outer", "length": 32},
                        {"iter_var": "rc.inner", "origin": "rc", "brother": "rc.outer", "length": 8}]}]

    accelerator = {"dataflow": "gemmini_ws",
                   "PEs": "256", "x": "16", "y": "16"}

    l, t, p, e, a = model.analyze(
        "tests2", "./tmp/"+"tests2"+"_"+"gemmini_ws"+".m", infos, accelerator)
    print("Finish modeling.")
    print("latency: " + str(l) + " throughput: " + str(t) +
          " power: " + str(p) + " energy: " + str(e) + " area: " + str(a))
