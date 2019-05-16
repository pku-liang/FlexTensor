import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="path to log file", type=str, default="")
    parser.add_argument("-t", "--target", help="path to target file", type=str, default="")
    args = parser.parse_args()
    points_lst = []
    with open(args.file, "r") as fin:
        for line in fin:
            if "Optimize" in line:
                points_lst.append([])
            if "The best currently" in line:
                perf = float(line.split(" ")[3])
                points_lst[-1].append(perf)

    with open(args.target, "a") as fout:
        for i, part in enumerate(points_lst):
            fout.write(str(i+1) + "\n")
            for point in part:
                fout.write(str(point) + "\n")
            fout.write("\n")


if __name__ == "__main__":
    main()