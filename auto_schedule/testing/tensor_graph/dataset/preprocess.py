import numpy as np


ratio = 0.8


if __name__ == "__main__":
    raw = []
    with open("all.txt", "r") as fin:
        for line in fin:
            raw.append(line)
    if not raw[-1][-1] == "\n":
        raw[-1] = raw[-1] + "\n"
    np.random.shuffle(raw)
    length = int(ratio * len(raw))
    train = raw[:length]
    test = raw[length:]
    with open("all_train.txt", "w") as fout:
        fout.writelines(train)
    with open("all_test.txt", "w") as fout:
        fout.writelines(test)
