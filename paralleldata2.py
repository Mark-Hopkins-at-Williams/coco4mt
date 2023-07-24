from datasets import Dataset, DatasetDict
from pandas import DataFrame

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


def nllb_data(src, tgt, lines=None):
    """Creates a DatasetDict with the NLLB data.
    
    If you provide a set of line numbers (as zero-indexed ints) for the
    lines argument, then this function will subselect only those lines
    for the training partition (the full validation and test sets are
    always used).    
    
    """
    
    data = []
    subcorpus = enumerate_lines(f'/mnt/storage/alexroot/coco4mt/nllb/data/{src}')
    target_corpus = enumerate_lines(f'/mnt/storage/alexroot/coco4mt/nllb/data/{tgt}')
    for i in range(len(subcorpus)):
        if lines is None or i in lines:
            if len(subcorpus[i]) > 0 and len(target_corpus[i]) > 0:
                item = {'id': i, 
                        'translation': {src: subcorpus[i],
                                        tgt: target_corpus[i]}}
                data.append(item)
    train_ds = Dataset.from_pandas(DataFrame(data=data))
    
    data = []
    subcorpus = enumerate_lines(f'/mnt/storage/alexroot/coco4mt/flores200_dataset/dev/{src}.dev')
    target_corpus = enumerate_lines(f'/mnt/storage/alexroot/coco4mt/flores200_dataset/dev/{tgt}.dev')
    for i in range(len(subcorpus)):
        if len(subcorpus[i]) > 0 and len(target_corpus[i]) > 0:
            item = {'id': i, 
                    'translation': {src: subcorpus[i],
                                    tgt: target_corpus[i]}}
            data.append(item)
    valid_ds = Dataset.from_pandas(DataFrame(data=data))
    
    data = []
    subcorpus = enumerate_lines(f'/mnt/storage/alexroot/coco4mt/flores200_dataset/devtest/{src}.devtest')
    target_corpus = enumerate_lines(f'/mnt/storage/alexroot/coco4mt/flores200_dataset/devtest/{tgt}.devtest')
    for i in range(len(subcorpus)):
        if len(subcorpus[i]) > 0 and len(target_corpus[i]) > 0:
            item = {'id': i, 
                    'translation': {src: subcorpus[i],
                                    tgt: target_corpus[i]}}
            data.append(item)
    test_ds = Dataset.from_pandas(DataFrame(data=data))
    result = DatasetDict()
    result['train'] = train_ds
    result['validation'] = valid_ds
    result['test'] = test_ds
    return result


