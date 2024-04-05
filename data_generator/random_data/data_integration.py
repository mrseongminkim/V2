import random
import configparser

config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")

random.seed(int(config["seed"]["integrate_data"]))

train_file = open("./data/random_data/train.csv", "w")
valid_file = open("./data/random_data/valid.csv", "w")

data = []
for alphabet_size in (2, 4, 6, 8, 10):
    file = open(f"./data/random_data/size_{alphabet_size}/train.csv", "r")
    data += file.readlines()

random.shuffle(data)

train_file.writelines(data[: int(len(data) * 0.9)])
valid_file.writelines(data[int(len(data) * 0.9) :])
