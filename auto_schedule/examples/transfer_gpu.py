import os
from auto_schedule.train import Entity, train_op_schedule_gpu_general_dx


arg_lst = [
        (1, 7, 7, 1024, 3, 3, 1024, 1, 1),
        # (8, 7, 7, 1024, 3, 3, 1024, 1, 1),
        # (64, 7, 7, 1024, 3, 3, 1024, 1, 1),
        # (256, 7, 7, 1024, 3, 3, 1024, 1, 1),
        (1, 14, 14, 1024, 1, 1, 512, 1, 0),
        (1, 28, 28, 256, 3, 3, 512, 1, 1),
        (1, 28, 28, 512, 1, 1, 256, 1, 0),
        (1, 56, 56, 128, 3, 3, 256, 1, 1),
        (1, 56, 56, 192, 1, 1, 128, 1, 0),
        (1, 112, 112, 64, 3, 3, 192, 1, 1),
        (1, 448, 448, 3, 7, 7, 64, 2, 3),
        (1, 1024, 1024, 1024),
    ]

names = [
        "yolo24_b1",
        # "yolo24_b8",
        # "yolo24_b64",
        # "yolo24_b256",
        "yolo19_b1",
        "yolo10_b1",
        "yolo7_b1",
        "yolo4_b1",
        "yolo3_b1",
        "yolo2_b1",
        "yolo1_b1",
        "gemm_1024"
    ]

func_names = ["conv2d_nchw"] * 8 + ["matmul_batch"]


def transfer(pre_train, post_train):
    entities = []
    for i in pre_train:
        args = arg_lst[i]
        entities.append(Entity(func_names[i], args))
    model_path = "gpu_transfer_pre{}_post{}.pkl".format(pre_train, post_train)
    model_path = os.path.abspath(model_path)
    train_op_schedule_gpu_general_dx(entities, 20, 50, model_path,
                                     logfile="process_transfer_pre_{}_(post{})_gpu.txt".format(pre_train, post_train),
                                     device="cuda:0")
    entities = []
    for i in post_train:
        args = arg_lst[i]
        entities.append(Entity(func_names[i], args))
    model_path = "gpu_transfer0_pre{}_post{}.pkl".format(pre_train, post_train)
    model_path = os.path.abspath(model_path)
    train_op_schedule_gpu_general_dx(entities, 20, 50, model_path,
                                     logfile="process_transfer_(pre_{})_post{}_gpu.txt".format(pre_train, post_train),
                                     device="cuda:0")


if __name__ == "__main__":
    transfer([1, 3], [5])
    transfer([0, 2], [4])
    transfer([0, 1, 2], [6])
    transfer([1, 3, 5], [8])
