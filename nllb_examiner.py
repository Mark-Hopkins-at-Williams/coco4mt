from pathlib import Path
from pathlib import PurePath

line_count = dict()
file_pair = dict()
common_list = list()
common_list_dict = dict()

# get all English files from NLLB-Seed
eng_files = sorted(Path('/home/data/NLLB-Seed').glob('**/eng_Latn'))

# populate the file_pair dict to match eng files to other lang files
for file in eng_files:
    file_pair[file] = list(Path(PurePath(file).parent).glob('./*'))
    file_pair[file].remove(file)
    file_pair[file] = file_pair[file][0]

# identify all lines that appear in all English files
for file in eng_files:
    with file.open() as file:
        for line in file:
            if line in line_count:
                line_count[line] += 1
            else:
                line_count[line] = 1
for sent in line_count.keys():
    appearances = line_count[sent]
    if appearances == 39:
        common_list.append(sent)

# map the lines common to all files to their locations in that list
common_list.sort()
for i, line in enumerate(common_list):
    common_list_dict[line] = i

# write the common eng lines into a file
with open(f'./nllb/data/eng_Latn', 'w') as writer:
    for line in common_list:
        writer.write(line)

# write revised files in each language to make parallel corpora
for file in eng_files:
    trans_file = file_pair[file]
    trans_name = PurePath(file_pair[file]).name
    new_trans_list = [" " for i in range(len(common_list))]
    map_to_common = dict()

    # map the locations of common lines in this eng file to their locations in the general eng file
    with file.open() as file:
        for i, line in enumerate(file):
            if line in common_list:
                map_to_common[i] = common_list_dict[line]

    # use those locations to order the corresponding lines in the translated file
    with trans_file.open() as trans_file:
        for i, line in enumerate(trans_file):
            if i in map_to_common.keys():
                new_trans_list[map_to_common[i]] = line
    
    # write a translated parallel file in the proper order
    with open(f'./nllb/data/{trans_name}', 'w') as writer:
        for line in new_trans_list:
            writer.write(line)
