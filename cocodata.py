

def load_coco_english(coco_eng_path, split):
    """
    Returns the specified split of the coco4mt English data as a list, with
    each sentence as an element.
    """
    sents = list()
    with open(f"{coco_eng_path}/{split}.txt") as reader:
        for line in reader:
            line = line.strip()
            sents.append(line)
    return sents