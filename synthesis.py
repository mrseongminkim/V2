import sys
import argparse
import time
import pickle
import signal
import configparser
import pathlib
from multiprocessing import Manager

# https://github.com/pytorch/pytorch/issues/3678
# This is why we only save state_dict
sys.path.insert(0, "./NeuralSplitter")

import torch
import numpy as np
import FAdo.reex as reex
import re2 as re

import NeuralSplitter.dataset as dataset
from NeuralSplitter.dataset import Vocabulary
from NeuralSplitter.checkpoint import Checkpoint
from NeuralSplitter.seed import seed_all
from NeuralSplitter.models import *

from submodels.SCNF.examples import Examples
from submodels.SCNF.util import *

from split import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    default="./data/practical_data/test_practicalregex.csv",
    dest="data_path",
    help="Path to data",
)
parser.add_argument("--log_path", default="./log_data/practical", dest="log_path", help="Path to save log data")
# batch size 기본 1이다. script에서도 안 건드린다.
parser.add_argument("--batch_size", action="store", dest="batch_size", help="batch size", default=1)
# best_accuracy 내에서 가장 최신 모델을 사용한다.
# 여러개 훈련시킬때 조심하자.
parser.add_argument(
    "--checkpoint_pos",
    default="./saved_models/practical/rnntype_gru_hidden_128/best_accuracy",
    dest="checkpoint_pos",
    help="path to checkpoint for splitting positive strings ",
)
parser.add_argument(
    "--sub_model",
    action="store",
    dest="sub_model",
    default="alpharegex",
    help="sub model used in generating sub regex from sub strings",
)
parser.add_argument(
    "--data_type",
    action="store",
    dest="data_type",
    default="practical",
    help="data type: random or practical",
)
parser.add_argument(
    "--alphabet_size",
    action="store",
    dest="alphabet_size",
    help="define the alphabet size of the regex",
    type=int,
    default=10,
)
parser.add_argument("--time_limit", action="store", dest="time_limit", help="time_limit", type=int, default=3)
parser.add_argument(
    "--synthesis_strategy",
    action="store",
    dest="synthesis_strategy",
    default="sequential_basic",
    help="synthesis_strategy: sequential_prefix, parallel, sequential_basic",
)
parser.add_argument("--exclude_GT", action="store_true", dest="exclude_GT", help="decide to not infer GT split")
parser.add_argument(
    "--exclude_Direct",
    action="store_true",
    dest="exclude_Direct",
    help="decide to not infer Direct split",
)
opt = parser.parse_args()

config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")
seed_all(int(config["seed"]["main"]))

COUNT_LIMIT = 1_000_000

if "regex_generator" in opt.sub_model:
    MAX_TIME_LIMIT = 15
else:
    MAX_TIME_LIMIT = opt.time_limit

if "random" in opt.data_type:
    MAX_SEQUENCE_LENGTH = 10
else:
    MAX_SEQUENCE_LENGTH = 15

if "blue_fringe" in opt.sub_model:
    membership_type = lambda regex, string: reex.str2regexp(regex).evalWordP(string)
else:
    membership_type = lambda regex, string: bool(re.fullmatch(regex, string))

# 기본적으로 sequential basic
use_prefix_every = False
if opt.synthesis_strategy == "sequential_basic":
    generate_regex_from_split = generate_split_regex_sequential
elif opt.synthesis_strategy == "parallel":
    generate_regex_from_split = generate_split_regex_in_parallel
elif opt.synthesis_strategy == "sequential_prefix":
    generate_regex_from_split = generate_split_regex_sequential
    use_prefix_every = True
else:
    raise Exception("unknown synthesis strategy")

# direct: w/o split
# dc: divide and conquer
# gt: ground truth
direct_time_total, dc_time_total, gt_time_total = 0, 0, 0
direct_correct_count, dc_correct_count, gt_correct_count = 0, 0, 0
direct_win, dc_win = 0, 0

manager = Manager()
# 서브 프로세스에서 공유되는 dict
return_dict = manager.dict()


class TimeOutException(Exception):
    pass


def alarm_handler(signum, frame):
    raise TimeOutException()


def convert_indices_to_strings(indices):
    # tensor: examples * max_len
    # strings may have examples less than indices' due to strings of only <pad>
    strings = []
    vocab = Vocabulary()
    for i in range(indices.size(0)):
        string = "".join([vocab.itos[x] for x in indices[i][indices[i] != vocab.stoi["<pad>"]]])
        if string:
            strings.append(string)
    return strings


def divide_and_conquer(model, pos, neg, pos_str, neg_str, idx):
    global dc_correct_count, dc_time_total
    start_time = time.time()
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(MAX_TIME_LIMIT)
    try:
        _, _, other = model(pos, MAX_SEQUENCE_LENGTH)
        # other[sequence]: list안에 max_len개의 example * 1 텐서가 있음
        # 첫번째 원소부터 time_step이며 텐서의 index는 nth examples
        # time_step * examples

        # pos: batch * examples * max_len
        splited_pos, sigma_list = split(pos, other["sequence"])  # batch, set, seq
        # print(splited_pos)
        # exit()
        splited_neg, _ = split(neg, other["sequence"], no_split=True)  # batch, set, seq
        # splitted_pos: 각 string을 substring으로 나눈다. List of batch * examples * splits
        # sigma_list: 각 substring이 sigma에서 나온것인지 표시한다. List of batch * examples * splits
        # splitted_neg: 각 string. List of batch * examples

        dc_answer, split_size = generate_regex_from_split(
            splited_pos[0],
            splited_neg[0],
            True,
            COUNT_LIMIT,
            alphabet_size=opt.alphabet_size,
            data_type=opt.data_type,
            sigma_lst=sigma_list,
            submodel=opt.sub_model,
            return_dict=return_dict,
            use_prefix_every=use_prefix_every,
        )
    except TimeOutException as e:
        print(e)
        print("time limit")
        dc_answer = None
    end_time = time.time()
    signal.alarm(0)
    if dc_answer is None:
        dc_correct = False
    else:
        try:
            dc_correct = is_solution(dc_answer, Examples(pos=pos_str, neg=neg_str), membership_type)
        except:
            dc_correct = False

    if dc_correct:
        dc_correct_count += 1
        dc_time_taken = end_time - start_time
    else:
        dc_time_taken = MAX_TIME_LIMIT
    dc_time_total += dc_time_taken

    # print(
    #    f"{idx}th Generated Regex (via DC): {dc_answer} ({dc_correct}), Time Taken: ",
    #    dc_time_taken,
    # )

    return dc_correct, dc_time_taken, dc_answer, other


def direct(other, pos, neg, pos_str, neg_str, idx):
    global direct_correct_count, direct_time_total
    start_time = time.time()
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(MAX_TIME_LIMIT)
    try:
        # _, _, other = pos_split_model(pos, None, regex)
        splited_pos, _ = split(pos, other["sequence"], no_split=True)  # batch, set, seq

        # _, _, other = neg_split_model(neg)
        splited_neg, _ = split(neg, other["sequence"], no_split=True)  # batch, set, seq

        direct_answer, split_size = generate_split_regex_sequential(
            splited_pos[0],
            splited_neg[0],
            False,
            COUNT_LIMIT,
            alphabet_size=opt.alphabet_size,
            data_type=opt.data_type,
            submodel=opt.sub_model,
            return_dict=return_dict,
            use_prefix_every=use_prefix_every,
        )
    except Exception as e:
        print("time limit")
        direct_answer = None
    end_time = time.time()
    signal.alarm(0)

    if direct_answer is None:
        direct_correct = False
    else:
        try:
            direct_correct = is_solution(direct_answer, Examples(pos=pos_str, neg=neg_str), membership_type)
        except:
            direct_correct = False

    if direct_correct:
        direct_correct_count += 1
        direct_time_taken = end_time - start_time
    else:
        direct_time_taken = MAX_TIME_LIMIT
    direct_time_total += direct_time_taken

    # print(
    #    f"{idx}th Generated Regex (direct): {direct_answer}, Time Taken: ",
    #    direct_time_taken,
    # )
    return direct_correct, direct_time_taken, direct_answer


def ground_truth(other, pos, neg, pos_str, neg_str, tag, idx):
    global gt_correct_count, gt_time_total
    # via ground truth -----------------------------------------------------------------
    start_time = time.time()
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(MAX_TIME_LIMIT)

    try:
        gt_answer = None

        splited_pos, sigma_lst = split(pos, np.array(tag, dtype="object").T)  # batch, set, seq

        # _, _, other = neg_split_model(neg)
        splited_neg, _ = split(neg, other["sequence"], no_split=True)  # batch, set, seq

        gt_split_size = len(splited_pos[0][0])

        gt_answer, split_size = generate_regex_from_split(
            splited_pos[0],
            splited_neg[0],
            True,
            COUNT_LIMIT,
            alphabet_size=opt.alphabet_size,
            data_type=opt.data_type,
            sigma_lst=sigma_lst,
            submodel=opt.sub_model,
            return_dict=return_dict,
            use_prefix_every=use_prefix_every,
        )
    except Exception as e:
        print(e)
        print("time limit")
    end_time = time.time()
    signal.alarm(0)

    if gt_answer is None:
        gt_correct = False
    else:
        try:
            gt_correct = is_solution(gt_answer, Examples(pos=pos_str, neg=neg_str), membership_type)
        except:
            gt_correct = False

    if gt_correct:
        gt_correct_count += 1
        gt_time_taken = end_time - start_time
    else:
        gt_time_taken = MAX_TIME_LIMIT
    gt_time_total += gt_time_taken

    # print(
    #    f"{idx}th Generated Regex (via GT): {gt_answer} ({gt_correct}), Time Taken: ",
    #    gt_time_taken,
    # )
    return gt_correct, gt_time_taken, gt_answer


# done
def synthesize_regex(model, pos, neg, pos_str, neg_str, tag, idx):
    direct_correct, direct_time_taken, direct_answer = None, None, None
    gt_correct, gt_time_taken, gt_answer = None, None, None

    dc_correct, dc_time_taken, dc_answer, other = divide_and_conquer(model, pos, neg, pos_str, neg_str, idx)
    if not opt.exclude_Direct:
        direct_correct, direct_time_taken, direct_answer = direct(other, pos, neg, pos_str, neg_str, idx)
    if not opt.exclude_GT:
        gt_correct, gt_time_taken, gt_answer = ground_truth(other, pos, neg, pos_str, neg_str, tag, idx)
    return (
        dc_correct,
        dc_time_taken,
        dc_answer,
        direct_correct,
        direct_time_taken,
        direct_answer,
        gt_correct,
        gt_time_taken,
        gt_answer,
    )


# done
def main():
    global dc_win, direct_win
    data = dataset.get_loader(
        opt.data_path,
        batch_size=opt.batch_size,
        is_test=True,
        shuffle=False,
        max_len=MAX_SEQUENCE_LENGTH,
    )

    pos_checkpoint = Checkpoint.load(Checkpoint.get_latest_checkpoint(opt.checkpoint_pos))
    pos_split_model = pos_checkpoint.model
    pos_split_model.eval()

    for count, tuple in enumerate(data):
        print(f"{count}/{len(data)}")
        # valid_pos, valid_neg doesn't need vector form
        pos, neg, subregex_list, valid_pos, valid_neg, label = tuple
        # blue_fringe cannot handle special character '_' and '!'
        # 이러면 ASCII 확장이 안 될텐데
        # _나 !이면 z로 바꾼다. -> doesn't hold anymore
        if opt.sub_model == "blue_fringe" and opt.data_type == "practical":
            pos = list(
                map(
                    lambda x: list(
                        map(
                            lambda y: torch.tensor([61]) if y.item() == 62 or y.item() == 63 else y,
                            x,
                        )
                    ),
                    pos,
                )
            )
            neg = list(
                map(
                    lambda x: list(
                        map(
                            lambda y: torch.tensor([61]) if y.item() == 62 or y.item() == 63 else y,
                            x,
                        )
                    ),
                    neg,
                )
            )
            valid_pos = list(
                map(
                    lambda x: list(
                        map(
                            lambda y: torch.tensor([61]) if y.item() == 62 or y.item() == 63 else y,
                            x,
                        )
                    ),
                    valid_pos,
                )
            )
            valid_neg = list(
                map(
                    lambda x: list(
                        map(
                            lambda y: torch.tensor([61]) if y.item() == 62 or y.item() == 63 else y,
                            x,
                        )
                    ),
                    valid_neg,
                )
            )

        pos_set = convert_indices_to_strings(pos[0])
        neg_set = convert_indices_to_strings(neg[0])
        valid_pos_set = convert_indices_to_strings(valid_pos[0])
        valid_neg_set = convert_indices_to_strings(valid_neg[0])

        # empty P set
        if not pos_set:
            continue

        # regex[0]인것으로 보아서 batch size를 늘릴 계획은 없었다.
        # regex = (batch * num_subregex)
        regex_string = "".join([f"({subregex})" for subregex in subregex_list[0]])

        # print("-" * 50)
        # print("Positive Strings:", ", ".join(pos_set))
        # print("Negative Strings:", ", ".join(neg_set))
        # print("Target Regex:", regex_string)
        # print("-" * 50)

        (
            dc_correct,
            dc_time_taken,
            dc_answer,
            direct_correct,
            direct_time_taken,
            direct_answer,
            gt_correct,
            gt_time_taken,
            gt_answer,
        ) = synthesize_regex(pos_split_model, pos, neg, pos_set, neg_set, label, count)

        # win rate
        if not opt.exclude_Direct:
            if dc_correct and direct_correct:
                if direct_time_taken > dc_time_taken:
                    dc_win += 1
                else:
                    direct_win += 1
            elif dc_correct:
                dc_win += 1
            elif direct_correct:
                direct_win += 1

        if not opt.exclude_Direct:
            print(f"Divide-and-conquer win rate over Direct: {dc_win / (dc_win + direct_win + 1e-9) * 100:.4f}%, Direct Total Time: {direct_time_total:.4f}, DC Total Time: {dc_time_total:.4f}")
            print(f"DC Success Ratio: {dc_correct_count / (count + 1) * 100:.4f}%, Direct Success Ratio: {direct_correct_count / (count + 1) * 100:.4f}%")
            print("-" * 50)
        else:
            print(f"DC Total Time: {dc_time_total:.4f}")
            print(f"DC Success Ratio: {dc_correct_count / (count + 1) * 100:.4f}%")
            print("-" * 50)

        log_data = dict()
        log_data["Target_string"] = regex_string
        log_data["pos"] = pos_set
        log_data["neg"] = neg_set
        log_data["pos_validation"] = valid_pos_set
        log_data["neg_validation"] = valid_neg_set
        log_data["DC_answer"] = dc_answer
        log_data["DC_success_ratio"] = dc_correct_count / (count + 1) * 100
        log_data["DC_time"] = dc_time_taken
        log_data["DC_total_time"] = dc_time_total

        if not opt.exclude_Direct:
            log_data["Direct_answer"] = direct_answer
            log_data["win_rate"] = dc_win / (dc_win + direct_win + 1e-9) * 100
            log_data["Direct_success_ratio"] = direct_correct_count / (count + 1) * 100
            log_data["Direct_time"] = direct_time_taken
            log_data["Direct_total_time"] = direct_time_total

        if not opt.exclude_GT:
            log_data["GT_answer"] = gt_answer
            log_data["GT_success_ratio"] = gt_correct_count / (count + 1) * 100
            log_data["GT_time"] = gt_time_taken
            log_data["GT_total_time"] = gt_time_total

        log_path = opt.log_path + "/{}".format(opt.sub_model)

        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
        with open(log_path + "/" + str(count) + ".pickle", "wb") as fw:
            pickle.dump(log_data, fw)


if __name__ == "__main__":
    main()
