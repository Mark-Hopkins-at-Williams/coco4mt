

def load_coco_english(coco_eng_path, split):
    """
    Returns the specified split of the coco4mt English data as a list, with
    each sentence as an element.

    Parameters
    ----------
    coco_eng_path : String
        the path to the eng folder of the coco4mt data in the local directory
    split : String
        the desired split of the English data (train, dev, or test)
    """
    sents = list()
    with open(f"{coco_eng_path}/{split}.txt") as reader:
        for line in reader:
            line = line.strip()
            sents.append(line)
    return sents