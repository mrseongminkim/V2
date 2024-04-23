import random
import configparser
import pathlib

config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")
random.seed(int(config["seed"]["integrate_data"]))

data_list = [
    "./data/practical_data/org/snort-clean.csv",
    "./data/practical_data/org/regexlib-clean.csv",
]

for i in range(1, 55):
    data_list.append(f"./data/practical_data/org/practical_regexes_{i:02}.csv")

pathlib.Path("./data/practical_data/integrated").mkdir(parents=True, exist_ok=True)
train_file = open("./data/practical_data/integrated/train.csv", "w")
valid_file = open("./data/practical_data/integrated/valid.csv", "w")
test_snort_file = open("data/practical_data/integrated/test_snort.csv", "w")
test_regexlib_file = open("data/practical_data/integrated/test_regexlib.csv", "w")
test_practical_file = open("data/practical_data/integrated/test_practicalregex.csv", "w")

practical = []
train = []
valid = []
intergrated_data = []
for data_idx, path in enumerate(data_list):
    file = open(path, "r")
    data = file.readlines()
    random.shuffle(data)

    if data_idx == 0:
        test_snort_file.writelines(data[:30])
        valid.extend(data[30:60])
        train.extend(data[60:])
    elif data_idx == 1:
        test_regexlib_file.writelines(data[:100])
        valid.extend(data[100:200])
        train.extend(data[200:])
    else:
        practical.extend(data)

test_practical_file.writelines(practical[:25_000])
valid.extend(practical[25_000:50_000])
train.extend(practical[50_000:])

random.shuffle(valid)
random.shuffle(train)

train_file.writelines(train)
valid_file.writelines(valid)
