import json
import multiprocessing as multi
from auto_schedule.train import get_compute_entities, generate_simple_train_data
from auto_schedule import SimpleEvaluator, ReverseBFSVisitor, ConstSpecifier


def generate_extractor_train_data(start=0, number=10, prefix="extractor_train_data", lower=False, q=None):
    entities = get_compute_entities(name="conv2d_channel_batch", target="llvm")
    evaluator = SimpleEvaluator()
    visitor = ReverseBFSVisitor()
    print("Generating data...")
    for i in range(number):
        print("    working on data ", i + start)
        specifier = ConstSpecifier()
        train_data = generate_simple_train_data(entities, specifier, evaluator, visitor)
        with open("{}_{}.txt".format(prefix, i + start), "w") as f:
            for data in train_data:
                json_obj = data.to_json()
                lower = str(data.get_lower()) + "\n\n"

                json_str = json.dumps(json_obj) + "\n"
                f.write(json_str)
                f.write(lower)
    print("Generation done!")
    if q:
        q.put(0)


def generate_extractor_train_data_multi(thread=10, start=0, number=10, prefix="extractor_train_data"):
    p = []
    q = multi.Queue()
    for i in range(thread):
        p.append(multi.Process(target=generate_extractor_train_data, args=(start + i * number, number, prefix, q)))
    for pr in p:
        pr.start()
    count = 0
    while count < thread:
        ret = q.get()
        count = count + 1
    for pr in p:
        pr.terminate()


def main():
    NUMBER = 1
    PREFIX = "extractor_train_data"
    generate_extractor_train_data(start=0, number=NUMBER, prefix=PREFIX)


if __name__ == "__main__":
    main()