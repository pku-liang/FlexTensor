import os


if __name__ == "__main__":
    dir_names = os.listdir(".")
    dir_names = list(filter(lambda x: os.path.isdir(x), dir_names))
    print(dir_names)
    for name in dir_names:
        print(name.split("conv"))
    write_lines = []
    dir_names = sorted(dir_names, key=lambda x: int(x.split("conv")[1]))
    for dir_name in dir_names:
        dir_path = os.path.join(".", dir_name)
        file_names = os.listdir(dir_path)
        for file_name in file_names:
            if "config" in file_name:
                file_path = os.path.join(dir_path, file_name)
                with open(file_path, "r") as fin:
                    lines = fin.readlines()
                    if lines:
                        line = lines[-1]
                        write_lines.append(line)
    with open("configs.txt", "w") as fout:
        for line in write_lines:
            fout.write(line) 