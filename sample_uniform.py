from random import sample
import sys


def uniform_sample(num_instances, budget=0.2):
    """Samples a specified percentage (budget) of training instances."""
    return sample(list(range(num_instances)), int(budget*num_instances))


if __name__ == "__main__":
    sentence_file = sys.argv[1]
    budget_pct = float(sys.argv[2])
    num_trials = int(sys.argv[3])
    sents = []
    with open(sentence_file) as reader:
        for line in reader:
            sents.append(line.strip())
    for i in range(num_trials):
        with open(f"usample.{i}.txt", 'w') as writer:
            selected_lines = sorted(uniform_sample(len(sents), budget_pct))
            for line in selected_lines:
                writer.write(f"{line}\n")