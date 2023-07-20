from datasets import Dataset, DatasetDict
from pandas import DataFrame

code3 = {
        "en": "eng",
        "de": "deu",
        "id": "ind",
        "ko": "kor",
        "fr": "fra",
        "my": "mya",
        "gu": "guj"
    }


def enumerate_lines(filename):
    lines = []
    with open(filename) as reader:
        for line in reader:
            line = line.strip() 
            lines.append(line)
    return lines        


def lines_to_exclude():
    result = set()
    with open('exclude.txt') as reader:
        for line in reader:
            result.add(int(line))
    return result


def coco_data(src, tgt, lines=None):
    """Creates a DatasetDict with the Coco4MT data.
    
    If you provide a set of line numbers (as zero-indexed ints) for the
    lines argument, then this function will subselect only those lines
    for the training partition (the full validation and test sets are
    always used).    
    
    """
    train_corpora = {'eng': enumerate_lines("/home/data/coco4mt-shared-task/hr_dataset/eng/train.txt"),
                     'deu': enumerate_lines("/home/data/coco4mt-shared-task/hr_dataset/deu/train.txt"),
                     'ind': enumerate_lines("/home/data/coco4mt-shared-task/hr_dataset/ind/train.txt"),
                     'kor': enumerate_lines("/home/data/coco4mt-shared-task/hr_dataset/kor/train.txt"),
                     'fra': enumerate_lines("/home/data/coco4mt-shared-task/lr_dataset/fra/train.txt"),
                     'mya': enumerate_lines("/home/data/coco4mt-shared-task/lr_dataset/mya/train.txt"),
                     'guj': enumerate_lines("/home/data/coco4mt-shared-task/lr_dataset/guj/train.txt")}
    valid_corpora = {'eng': enumerate_lines("/home/data/coco4mt-shared-task/hr_dataset/eng/dev.txt"),
                     'deu': enumerate_lines("/home/data/coco4mt-shared-task/hr_dataset/deu/dev.txt"),
                     'ind': enumerate_lines("/home/data/coco4mt-shared-task/hr_dataset/ind/dev.txt"),
                     'kor': enumerate_lines("/home/data/coco4mt-shared-task/hr_dataset/kor/dev.txt"),
                     'fra': enumerate_lines("/home/data/coco4mt-shared-task/lr_dataset/fra/dev.txt"),
                     'mya': enumerate_lines("/home/data/coco4mt-shared-task/lr_dataset/mya/dev.txt"),
                     'guj': enumerate_lines("/home/data/coco4mt-shared-task/lr_dataset/guj/dev.txt")}
    test_corpora  = {'eng': enumerate_lines("/home/data/coco4mt-shared-task/hr_dataset/eng/test.txt"),
                     'deu': enumerate_lines("/home/data/coco4mt-shared-task/hr_dataset/deu/test.txt"),
                     'ind': enumerate_lines("/home/data/coco4mt-shared-task/hr_dataset/ind/test.txt"),
                     'kor': enumerate_lines("/home/data/coco4mt-shared-task/hr_dataset/kor/test.txt"),
                     'fra': enumerate_lines("/home/data/coco4mt-shared-task/lr_dataset/fra/test.txt"),
                     'mya': enumerate_lines("/home/data/coco4mt-shared-task/lr_dataset/mya/test.txt"),
                     'guj': enumerate_lines("/home/data/coco4mt-shared-task/lr_dataset/guj/test.txt")}
    
    data = []
    corpus = train_corpora
    for i in range(len(corpus[code3[src]])):
        if lines is None or i in lines:
            if len(corpus[code3[src]][i]) > 0 and len(corpus[code3[tgt]][i]) > 0:
                item = {'id': i, 
                        'translation': {code3[src]: corpus[code3[src]][i],
                                        code3[tgt]: corpus[code3[tgt]][i]}}
                data.append(item)
    train_ds = Dataset.from_pandas(DataFrame(data=data))
    
    data = []
    corpus = valid_corpora
    for i in range(len(corpus[code3[src]])):
        if len(corpus[code3[src]][i]) > 0 and len(corpus[code3[tgt]][i]) > 0:
            item = {'id': i, 
                    'translation': {code3[src]: corpus[code3[src]][i],
                                    code3[tgt]: corpus[code3[tgt]][i]}}
            data.append(item)
    valid_ds = Dataset.from_pandas(DataFrame(data=data))
    
    data = []
    corpus = test_corpora
    for i in range(len(corpus[code3[src]])):
        if len(corpus[code3[src]][i]) > 0 and len(corpus[code3[tgt]][i]) > 0:
            item = {'id': i, 
                    'translation': {code3[src]: corpus[code3[src]][i],
                                    code3[tgt]: corpus[code3[tgt]][i]}}
            data.append(item)
    test_ds = Dataset.from_pandas(DataFrame(data=data))
    result = DatasetDict()
    result['train'] = train_ds
    result['validation'] = valid_ds
    result['test'] = test_ds
    return result


