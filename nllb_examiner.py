from pathlib import Path

line_count = dict()

files = sorted(Path('/home/data/NLLB-Seed').glob('**/eng_Latn'))

for file in files:
    with file.open() as file:
        for line in file:
            if line in line_count:
                line_count[line] += 1
            else:
                line_count[line] = 1

count_all = 0
count_not = 0
count_1 = 0

for sent in line_count.keys():
    appearances = line_count[sent]
    if appearances == 39:
        count_all += 1
    else:
        count_not += 1
        if appearances == 1:
            count_1 += 1

print(str(count_all) + " lines appear in all 39 files.")
print(str(count_not) + " lines do not appear in all 39 files.")
print(str(count_1) + " of those lines appear in only one file.")
print("There are " + str(count_all + count_not) + " distinct lines in all files. Each file has 6193 lines.")