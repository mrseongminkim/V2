from multiprocessing import Process, Manager
from collections import Counter

import torch

from submodels.RegexGenerator.batch import *
import submodels.SCNF.synthesizer
from submodels.SCNF.parsetree import *

import submodels.SCNF.synthesizer_snort
from submodels.SCNF.util_snort import *

from NeuralSplitter.dataset import Vocabulary
from submodels.SCNF.examples import Examples
from rpni import synthesis as rpni_synthesis


class TimeOutException(Exception):
    pass


class Ex:
    def __init__(self, pos, neg):
        self.pos = pos
        self.neg = neg

    def __str__(self):
        print(self.pos, self.neg)


def is_last_sigma(lst, split_size):
    # 애초에 이거 leftmost라서 안 먹힘;;;
    try:
        # idx = len(lst) - 1: 마지막 원소
        # list(reversed(lst)).index(split_size) 마지막 subregex의 인덱스 left most가 반환됨
        idx = len(lst) - 1 - list(reversed(lst)).index(split_size)
    except:
        return False
    # 이건 random을 기준으로 만들어진 것 같다.
    if idx != 9 and lst[idx + 1] == 0:
        return True


org2RG = {
    "A": "\.",
    "B": ":",
    "C": ",",
    "D": ";",
    "E": "_",
    "F": "=",
    "G": "[",
    "H": "]",
    "I": "/",
    "J": "\?",
    "K": "\!",
    "L": "\{",
    "M": "\}",
    "N": "\(",
    "O": "\)",
    "P": "\<",
}
RG2org = {v: k for k, v in org2RG.items()}


# change original string to RG formed string
def get_org2RG(string):
    for k, v in org2RG.items():
        string = string.replace(k, v)
    return string


def get_RG2org(string):
    for k, v in RG2org.items():
        string = string.replace(k, v)
    return string


def split(strings, label, no_split=False):
    vocab = Vocabulary()
    splited_string = []
    # 그냥 padding이나 unk만 없애줌
    if no_split:
        # strings.max라고 하면 64(pad) 아니면 65(unk)인데 pad라고 보고 이를 처리하는 것이라고 명시해주는 것이 좋을듯
        for batch_idx in range(len(strings)):
            set = []
            for set_idx in range(10):
                seq = []
                seq.append(
                    "".join(
                        map(
                            lambda x: vocab.itos[x],
                            [
                                x
                                for x in strings[batch_idx, set_idx]
                                if x != vocab.stoi["<pad>"] and x != vocab.stoi["<unk>"]
                            ],
                        )
                    )
                )
                set.append(seq)
            splited_string.append(set)
        return splited_string, None

    label = [i.tolist() for i in label]
    tmp = torch.LongTensor(label).transpose(0, 1).squeeze(-1).tolist()
    # examples * max_len
    # 각각이 어느 subregex에 속하는지 표시한다.

    # 서브 레겍스가 몇 개 존재하는지 확인하기 위해서 <pad>를 제외한 가장 큰 인덱스를 찾는다.
    # unique한 원소의 개수를 세는 것이 더 정확할 것 같습니다 ㅎㅎ
    split_size = torch.tensor(label)[torch.tensor(label) != vocab.stoi["<pad>"]].max().item()

    # x: 하나의 max_len sequence; 하나의 example
    # split_size: 몇 개의 regex가 있는지 (잘못됨)
    # 맨 마지막이 sigma star면 0이니까 카운터가 증가하지 않는다. -> 하나를 증가시켜준다.
    # 이 코드 자체도 문제가 있다.
    # 뭐 어쨌든... 이런 상황이 안 흔하니까....
    # 일단 보자
    if any(map(lambda x: is_last_sigma(x, split_size), tmp)):
        split_size += 1

    label2 = []
    sigma_lst = []
    # 각 example을 본다.
    # label2 몇번째 레겍스에서 나온건지 알려준다.
    # 정확히는 0번에 대해서 정규화를 진행해줌; 0번의 위치?
    for templete in tmp:
        tmp2 = []
        sigma_lst2 = []
        now = 0
        # 각 subregex index를 본다.
        for element in templete:
            if element != 0:
                if now != element and element != vocab.stoi["<pad>"]:
                    for _ in range(element - len(sigma_lst2)):
                        sigma_lst2.append(False)
                tmp2.append(element)
                now = element
            else:
                # sigma star로 생성됨
                # 이건 뭘까?
                # 빈 리스트거나 마지막이 False
                # 즉 연속된 0을 처리해주는 것
                # 하나의 연속된 0은 하나로 처리한다.
                if not sigma_lst2 or not sigma_lst2[-1]:
                    sigma_lst2.append(True)
                # now가 증가하지 않고 계속 넣기만 함
                tmp2.append(now + 1)

        while len(sigma_lst2) < split_size:
            sigma_lst2.append(False)

        label2.append(tmp2)
        sigma_lst.append(sigma_lst2)

    predict_dict = [dict(Counter(l)) for l in label2]
    # 각 서브 레겍스에서 얼마나 많은 글자들이 할당되어있는지
    for batch_idx in range(len(strings)):
        set = []
        for set_idx in range(10):
            src_seq = strings[batch_idx, set_idx].tolist()  # list of max_len
            # 배치 없으니까 set_idx를 가져오는 것과 동일
            predict_seq_dict = predict_dict[batch_idx * 10 + set_idx]
            seq = []
            idx = 0
            # subregex가 1부터 매핑되고 sigma star도 이렇게 처리를 해주었으니
            for seq_id in range(1, split_size + 1):
                tmp = ""
                if seq_id in predict_seq_dict.keys():
                    # 여기 float은 왜 넣은거야...?
                    for _ in range(predict_seq_dict[seq_id]):
                        tmp += vocab.itos[src_seq[idx]]
                        idx += 1
                seq.append(tmp)
            set.append(seq)
        splited_string.append(set)
    # 즉, 각 서브레겍스에 맞춰서 스트링을 잘라줌
    return splited_string, sigma_lst


def is_satisfy_pos(regex, examples, membership):
    for string in examples.getPos():
        if not membership(regex, string):
            return False
    return True


def get_sigma(examples):
    if is_satisfy_pos("\d*", examples, membership):
        return r"\d*"
    elif is_satisfy_pos("\w*", examples, membership):
        return r"\w*"
    else:
        return r".*"


# for subregex synthesis with baselines
def generate_regex_with_split_ar(
    sigma_lst,
    sub_id,
    sub_pos_set,
    sub_neg_set,
    split_model,
    count_limit,
    prefix,
    alphabet_size,
    data_type,
    return_dict,
):
    # 하나라면 이게 곧 레겍스이다.
    if len(sub_pos_set) == 1:
        return_dict[sub_id] = sub_pos_set.pop()
        return

    # divide and conquer에서는 start_with_no_concat=split_model이 True이다.

    if data_type == "random":
        # 각 example에서 sub_id에 해당하는 부분이 sigma인지 확인한다.
        # any인 이유는 이 부분을 사용하지 않고 넘어갔다면 False로 표현되기 때문이다.
        if sigma_lst is not None and any(list(map(lambda x: x[sub_id], sigma_lst))):
            # alpha regex를 돌리지 않고 바로 생성한다.
            tmp = repr(KleenStar(Or(*[Character(str(x)) for x in range(alphabet_size)])))
        else:
            tmp = repr(
                submodels.SCNF.synthesizer.synthesis(
                    Examples(pos=sub_pos_set, neg=sub_neg_set),
                    count_limit,
                    start_with_no_concat=split_model,
                    prefix_for_neg_test=prefix,
                    suffix_for_neg_test=None,
                    alphabet_size=alphabet_size,
                )
            )
    # practical
    else:
        if sigma_lst is not None and any(list(map(lambda x: x[sub_id], sigma_lst))):
            # 가장 작은 character class
            tmp = get_sigma(Examples(pos=sub_pos_set, neg=sub_neg_set))
        else:
            tmp, _ = submodels.SCNF.synthesizer_snort.synthesis(
                Examples(pos=sub_pos_set, neg=sub_neg_set),
                count_limit,
                start_with_no_concat=split_model,
                prefix_for_neg_test=prefix,
                suffix_for_neg_test=None,
                alphabet_size=alphabet_size,
            )
            tmp = repr(tmp)
    return_dict[sub_id] = tmp


def generate_regex_with_split_bf(
    sub_id, sub_pos_set, sub_neg_set, split_model, count_limit, prefix, alphabet_size, return_dict
):

    if len(sub_pos_set) == 1:
        return_dict[sub_id] = sub_pos_set.pop()
        return

    tmp = rpni_synthesis(
        Examples(pos=sub_pos_set, neg=sub_neg_set),
        count_limit,
        start_with_no_concat=split_model,
        prefix_for_neg_test=prefix,
        suffix_for_neg_test=None,
        alphabet_size=alphabet_size,
    )

    return_dict[sub_id] = str(tmp)


def generate_regex_with_split_rg(sigma_lst, sub_id, sub_pos_set, sub_neg_set, return_dict):

    if len(sub_pos_set) == 1:
        return_dict[sub_id] = sub_pos_set.pop()
        return

    # print(sub_pos_set, sub_neg_set)
    # new_pos_set = set()
    # for x in sub_pos_set:
    #     new_pos_set.add(get_org2RG(x))
    # new_neg_set = set()
    # for x in sub_neg_set:
    #     new_neg_set.add(get_org2RG(x))
    # print(new_pos_set, new_neg_set)

    if sigma_lst is not None and any(list(map(lambda x: x[sub_id], sigma_lst))):
        tmp = get_sigma(Examples(pos=sub_pos_set, neg=sub_neg_set))
    else:
        tmp = execute([Ex(list(sub_pos_set), list(sub_neg_set))])

    tmp = str(tmp).replace("++", "+").replace("?+", "+")

    # tmp = get_RG2org(tmp)

    return_dict[sub_id] = tmp


def generate_split_regex_sequential(
    splited_pos,
    splited_neg,
    split_model=False,
    count_limit=1000,
    alphabet_size=5,
    data_type="random",
    sigma_lst=None,
    submodel="alpharegex",
    return_dict=None,
    use_prefix_every=False,
):
    # splited_pos: 10 * substring
    # splited_neg: 10 * string
    split_size = len(splited_pos[0])
    print("Split Size: ", split_size)

    # 빈 example을 찾는다...?
    splited_pos = list(filter(lambda x: any(x), splited_pos))
    splited_neg = list(filter(lambda x: any(x), splited_neg))

    split_set = []

    for sub_id in range(split_size):
        pos = []
        neg = []

        for set_idx in range(len(splited_pos)):
            pos.append(splited_pos[set_idx][sub_id])
        # 이거 밖으로 빼내야지 불필요한 연산 안 한다.
        for set_idx in range(len(splited_neg)):
            neg.append(splited_neg[set_idx][0])
        if not neg:
            neg.append("")
        # pos: substring for each subregex
        # neg: every neg examples
        split_set.append([set(pos), set(neg)])

    # synthesis one by one
    for sub_id in range(split_size):
        # prefix strategy (only nth element or every element)
        # 마지막 원소거나 매번 프리픽스를 사용한다.
        if sub_id != 0 and (sub_id == split_size - 1 or use_prefix_every):
            # 전 prefix들을 ()로 감싼다.
            prefix = "(" + ")(".join([return_dict[i] for i in range(sub_id)]) + ")"
        else:
            # neg와 pos가 겹치면 제외한다.
            split_set[sub_id][1] -= split_set[sub_id][0]
            prefix = None

        print("Splited Positive Strings:", split_set[sub_id][0])
        print("Splited Negative Strings:", split_set[sub_id][1])

        if submodel == "alpharegex":
            generate_regex_with_split_ar(
                sigma_lst,
                sub_id,
                split_set[sub_id][0],
                split_set[sub_id][1],
                split_model,
                count_limit,
                prefix,
                alphabet_size,
                data_type,
                return_dict,
            )
        elif submodel == "blue_fringe":
            count_limit = 1000000000
            generate_regex_with_split_bf(
                sub_id,
                split_set[sub_id][0],
                split_set[sub_id][1],
                split_model,
                count_limit,
                prefix,
                alphabet_size,
                return_dict,
            )
        elif submodel == "regex_generator":
            generate_regex_with_split_rg(
                sigma_lst, sub_id, split_set[sub_id][0], split_set[sub_id][1], return_dict
            )
        else:
            raise Exception("unknown baseline")

    return "(" + ")(".join([return_dict[i] for i in range(split_size)]) + ")", split_size


def generate_split_regex_in_parallel(
    splited_pos,
    splited_neg,
    split_model=False,
    count_limit=1000,
    alphabet_size=5,
    data_type="random",
    sigma_lst=None,
    submodel="alpharegex",
    return_dict=None,
    use_prefix_every=False,
):

    split_size = len(splited_pos[0])
    print("Split Size: ", split_size)

    splited_pos = list(filter(lambda x: any(x), splited_pos))
    splited_neg = list(filter(lambda x: any(x), splited_neg))

    split_set = []
    procs = []

    for sub_id in range(split_size):
        pos = []
        neg = []

        for set_idx in range(len(splited_pos)):
            pos.append(splited_pos[set_idx][sub_id])
        for set_idx in range(len(splited_neg)):
            neg.append(splited_neg[set_idx][0])
        if not neg:
            neg.append("")

        split_set.append([set(pos), set(neg)])

    # parallel for regex_generator
    try:
        if submodel == "regex_generator":
            for sub_id in range(split_size):
                proc = Process(
                    target=generate_regex_with_split_rg,
                    args=(
                        sigma_lst,
                        sub_id,
                        split_set[sub_id][0],
                        split_set[sub_id][1],
                        return_dict,
                    ),
                )

                procs.append(proc)
                proc.start()

            for proc in procs:
                proc.join()

            return "(" + ")(".join([return_dict[i] for i in range(split_size)]) + ")"
    except Exception as e:
        for proc in procs:
            proc.terminate()
        raise TimeOutException()

    # parallel synthesis [1, n-1]
    try:
        prefix = None
        for sub_id in range(split_size - 1):
            if submodel == "alpharegex":
                proc = Process(
                    target=generate_regex_with_split_ar,
                    args=(
                        sigma_lst,
                        sub_id,
                        split_set[sub_id][0],
                        split_set[sub_id][1],
                        split_model,
                        count_limit,
                        prefix,
                        alphabet_size,
                        data_type,
                        return_dict,
                    ),
                )
            elif submodel == "blue_fringe":
                count_limit = 1000000000
                proc = Process(
                    target=generate_regex_with_split_bf,
                    args=(
                        sub_id,
                        split_set[sub_id][0],
                        split_set[sub_id][1],
                        split_model,
                        count_limit,
                        prefix,
                        alphabet_size,
                        return_dict,
                    ),
                )
            else:
                raise Exception("unknown baseline")

            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()
    except Exception as e:
        for proc in procs:
            proc.terminate()
        print("catch processes")
        raise TimeOutException()

    # synthesis for nth subregex
    if split_size > 1:
        prefix = "(" + ")(".join([return_dict[i] for i in range(split_size - 1)]) + ")"
    else:
        prefix = None

    if submodel == "alpharegex":
        generate_regex_with_split_ar(
            sigma_lst,
            split_size - 1,
            split_set[split_size - 1][0],
            split_set[split_size - 1][1],
            split_model,
            count_limit,
            prefix,
            alphabet_size,
            data_type,
            return_dict,
        )
    elif submodel == "blue_fringe":
        count_limit = 1000000000
        generate_regex_with_split_bf(
            split_size - 1,
            split_set[split_size - 1][0],
            split_set[split_size - 1][1],
            split_model,
            count_limit,
            prefix,
            alphabet_size,
            return_dict,
        )
    else:
        raise Exception("unknown baseline")

    return "(" + ")(".join([return_dict[i] for i in range(split_size)]) + ")", split_size
