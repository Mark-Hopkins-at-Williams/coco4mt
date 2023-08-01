

def load_coco_english(split):
    """
    Returns the specified split of the coco4mt English data as a list, with
    each sentence as an element.
    """
    sents = list()
    with open(f"/home/data/coco4mt-shared-task/hr_dataset/eng/{split}.txt") as reader:
        for line in reader:
            line = line.strip()
            sents.append(line)
    return sents