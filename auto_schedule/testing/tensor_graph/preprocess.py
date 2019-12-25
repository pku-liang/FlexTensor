import numpy as np


ratio = 0.8


if __name__ == "__main__":
    raw = []
    with open("data.txt", "r") as fin:
        for line in fin:
            raw.append(line)
    np.random.shuffle(raw)
    length = int(ratio * len(raw))
    train = raw[:length]
    test = raw[length:]
    with open("train.txt", "w") as fout:
        fout.writelines(train)
    with open("test.txt", "w") as fout:
        fout.writelines(test)