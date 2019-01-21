import json
from auto_schedule.train import get_compute_entities, generate_robust_train_data
from auto_schedule import SimpleEvaluator, ReverseBFSVisitor, RandomSpecifier, ConstSpecifier


def generate_case(name, number=3, prefix="case_study_", epoch=10, trial=3):
    entities = get_compute_entities(name=name, target="llvm")
    evaluator = SimpleEvaluator()
    visitor = ReverseBFSVisitor()
    print("Begin...")
    for i in range(number):
        print("    trial ", i)
        specifier = ConstSpecifier()
        train_data = generate_robust_train_data(entities, specifier, evaluator, visitor, epoch=epoch, trial=trial)
        with open("{}{}_{}.txt".format(prefix, name, i), "w") as f:
            for data in train_data:
                json_obj = data.to_json()
                lower = str(data.get_lower()) + "\n\n"

                json_str = json.dumps(json_obj) + "\n"
                f.write(json_str)
                f.write(lower)
    print("Done!")


if __name__ == "__main__":
    generate_case("conv2d_channel_batch", number=1, epoch=1, trial=100)
