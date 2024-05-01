import pickle
import argparse
import math
import os

import re2 as re
import FAdo.reex as reex
from FAdo.fa import *
from FAdo.cfg import *

from str2regexp import *

parser = argparse.ArgumentParser()
parser.add_argument("--path", dest="path", help="Path to experiment result directory.")
parser.add_argument("--time_limit", dest="time_limit", type=int, default=3, help="time limit")
parser.add_argument("--num", dest="num", type=int, default=1000, help="number of examples")
parser.add_argument("--exclude_GT", action="store_true", dest="exclude_GT", help="decide to not infer GT split")
parser.add_argument(
    "--exclude_Direct",
    action="store_true",
    dest="exclude_Direct",
    help="decide to not infer Direct split",
)
opt = parser.parse_args()


def membership(regex, string):
    return bool(re.fullmatch(regex, string))


def membership2(regex, string):
    return str2regexp(regex).evalWordP(string)


def confusion_matrix(answer, pos, neg):
    matrix = {"Tp": 0, "Tn": 0, "Fp": 0, "Fn": 0}
    for string in pos:
        if membership(answer, string):
            matrix["Tp"] = matrix["Tp"] + 1
        else:
            matrix["Fp"] = matrix["Fp"] + 1
    for string in neg:
        if membership(answer, string):
            matrix["Fn"] = matrix["Fn"] + 1
        else:
            matrix["Tn"] = matrix["Tn"] + 1
    return matrix


def MCC_score(matrix):
    # range: -1 to 1
    if (matrix["Tp"] + matrix["Fn"]) == 0:
        return -1
    elif (matrix["Tn"] + matrix["Fp"]) == 0:
        return 1
    else:
        return (matrix["Tp"] * matrix["Tn"] - matrix["Fp"] * matrix["Fn"]) / math.sqrt((matrix["Tp"] + matrix["Fp"]) * (matrix["Tn"] + matrix["Fn"]) * (matrix["Tp"] + matrix["Fn"]) * (matrix["Tn"] + matrix["Fp"]))


"""obsoleted
def full_log(num):
    with open(path + "/" + file_list[num], "rb") as fr:
        log_data = pickle.load(fr)
        print("-" * 50)
        print("Positive Strings:", ", ".join(log_data["pos"]))
        print("Negative Strings:", ", ".join(log_data["neg"]))
        print("Target Regex:", "".join(log_data["Target_string"]))
        print("-" * 50)
        print(
            f"{num}th Generated Regex (via DC): "
            + str(log_data["DC_answer"])
            + "("
            + str(log_data["DC_time"] < 20)
            + "), Time Taken: "
            + str(log_data["DC_time"])
        )
        print(
            f'{num}th Generated Regex (direct): {log_data["Direct_answer"]}, Time Taken: '
            + str(log_data["Direct_time"])
        )
        print(
            f'Divide-and-conquer win rate over Direct: {log_data["win_rate"]:.4f}%, Direct Total Time: {log_data["Direct_total_time"]:.4f}, DC Total Time: {log_data["DC_total_time"]:.4f}'
        )
        print(
            f'DC Success Ratio: {log_data["DC_success_ratio"]:.4f}%, Direct Success Ratio: {log_data["Direct_success_ratio"]:.4f}%'
        )
        print(log_data["pos_validation"])
        print(log_data["neg_validation"])
        print("-" * 50)
"""

if __name__ == "__main__":
    path = opt.path
    time_limit = opt.time_limit
    num = opt.num

    file_list = sorted(os.listdir(path), key=lambda x: int(re.sub("\D*", "", x)))

    print("path:", path)
    print("time limit:", time_limit)
    print("length of data:", str(len(file_list)))
    print("length of inferred data:", str(num))

    if "blue_fringe" in path:
        membership = membership2

    def membershipAll(regex, pos, neg):
        if not regex:
            return False
        for p in pos:
            if not membership(regex, p):
                return False
        for n in neg:
            if membership(regex, n):
                return False
        return True

    Direct_success_count = 0
    DC_success_count = 0
    GT_success_count = 0

    Direct_full_success_count, DC_full_success_count, GT_full_success_count = 0, 0, 0
    Direct_total_MCC, DC_total_MCC, GT_total_MCC = 0, 0, 0
    Direct_total_time, DC_total_time, GT_total_time = 0, 0, 0
    Direct_total_time_onlysucc, DC_total_time_onlysucc = 0, 0
    Direct_win, DC_win = 0, 0
    count_both = 0

    for idx, file_name in enumerate(file_list[:num]):
        with open(path + "/" + file_name, "rb") as fr:
            log_data = pickle.load(fr)

            """
            if log_data["DC_answer"] and log_data["GT_answer"]:
                if log_data["DC_answer"] != log_data["GT_answer"]:
                    print(log_data["DC_answer"])
                    print(log_data["GT_answer"])
            """
            # Direct success ratio ---------------------------
            if not opt.exclude_Direct:
                if log_data["Direct_time"] < time_limit and log_data["Direct_answer"]:
                    Direct_result = True
                    Direct_total_time += log_data["Direct_time"]
                    Direct_success_count += 1
                else:
                    Direct_result = False
                    Direct_total_time += time_limit

                # Direct scoring
                if Direct_result:
                    matrix = confusion_matrix(
                        log_data["Direct_answer"],
                        log_data["pos_validation"],
                        log_data["neg_validation"],
                    )
                    score = MCC_score(matrix)
                else:
                    score = -1

                Direct_total_MCC += score
                if score == 1:
                    Direct_full_success_count += 1

            # DC  ------------------------------
            # 예제에 대해서 생성이 되었는지 먼저 확인한다.
            if log_data["DC_time"] < time_limit and log_data["DC_answer"]:
                DC_result = True
                DC_total_time += log_data["DC_time"]
                DC_success_count += 1
            else:
                DC_result = False
                DC_total_time += time_limit

            # DC scoring
            if DC_result:
                matrix = confusion_matrix(log_data["DC_answer"], log_data["pos_validation"], log_data["neg_validation"])
                score = MCC_score(matrix)
            else:
                score = -1

            dc_score = score

            DC_total_MCC += score
            if score == 1:
                DC_full_success_count += 1

            # GT  ------------------------------
            if not opt.exclude_GT:
                if log_data["GT_time"] < time_limit and log_data["GT_answer"]:
                    GT_result = True
                    GT_total_time += log_data["GT_time"]
                    GT_success_count += 1
                else:
                    GT_result = False
                    GT_total_time += time_limit

                # GT scoring
                if GT_result:
                    matrix = confusion_matrix(
                        log_data["GT_answer"],
                        log_data["pos_validation"],
                        log_data["neg_validation"],
                    )
                    score = MCC_score(matrix)
                else:
                    score = -1

                gt_score = score

                GT_total_MCC += score
                if score == 1:
                    GT_full_success_count += 1

            """
            if dc_score > gt_score:
                print(log_data["DC_answer"])
                print(log_data["GT_answer"])
                print(log_data["pos_validation"])
                print(log_data["neg_validation"])
            """

            # Both; dc와 direct의 비교
            if not opt.exclude_Direct:
                if Direct_result and DC_result:
                    Direct_total_time_onlysucc += log_data["Direct_time"]
                    DC_total_time_onlysucc += log_data["DC_time"]
                    if log_data["Direct_time"] > log_data["DC_time"]:
                        DC_win += 1
                    else:
                        Direct_win += 1
                    count_both += 1
                elif Direct_result:
                    Direct_win += 1
                else:
                    DC_win += 1
    print("direct, dc, gt order")
    print(
        "Success ratio : ",
        str(Direct_success_count / num),
        str(DC_success_count / num),
        str(GT_success_count / num),
    )
    print(
        "Full success ratio : ",
        str(Direct_full_success_count / num),
        str(DC_full_success_count / num),
        str(GT_full_success_count / num),
    )
    print(
        "MCC score avg : ",
        str(Direct_total_MCC / num),
        str(DC_total_MCC / num),
        str(GT_total_MCC / num),
    )
    print("direct / dc")
    print("Win ratio:", str(Direct_win / (Direct_win + DC_win)), str(DC_win / (Direct_win + DC_win)))
    print("Run time avg:", str(Direct_total_time / num), str(DC_total_time / num))
    print(
        "Run time (only succ) avg:",
        str(Direct_total_time_onlysucc / count_both),
        str(DC_total_time_onlysucc / count_both),
    )

    print()
