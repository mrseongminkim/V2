from multiprocessing import Process
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

from str2regexp import *

from set2regex.checkpoint import Checkpoint
from set2regex.models import Seq2seq, EncoderRNN, DecoderRNN, attention

vocab = Vocabulary()

random_encoder = EncoderRNN(
    vocab_size=len(vocab),
    max_len=10,
    hidden_size=256,
    dropout_p=0,
    input_dropout_p=0,
    bidirectional=True,
    n_layers=2,
    rnn_cell="lstm",
    variable_lengths=False,
    set_transformer=False,
)
practical_encoder = EncoderRNN(
    vocab_size=len(vocab),
    max_len=15,
    hidden_size=256,
    dropout_p=0,
    input_dropout_p=0,
    bidirectional=True,
    n_layers=2,
    rnn_cell="lstm",
    variable_lengths=False,
    set_transformer=False,
)
atten = attention.Attention(256 * 2, True)
decoder = DecoderRNN(
    vocab_size=len(vocab),
    max_len=100,
    hidden_size=256 * 2,
    dropout_p=0,
    input_dropout_p=0,
    use_attention=atten,
    bidirectional=True,
    n_layers=2,
    rnn_cell="lstm",
    attn_mode=True,
)

# random_pt = Checkpoint.load(Checkpoint.get_latest_checkpoint("./saved_models/set2regex/random/lstm_256_2_True"))
# random_set2regex = Seq2seq(random_encoder, decoder)
# random_set2regex = random_set2regex.cuda()
# random_set2regex.load_state_dict(random_pt)
# random_set2regex.eval()

practical_pt = Checkpoint.load(Checkpoint.get_latest_checkpoint("./saved_models/set2regex/practical/lstm_256_2_True"))
practical_set2regex = Seq2seq(practical_encoder, decoder)
practical_set2regex = practical_set2regex.cuda()
practical_set2regex.load_state_dict(practical_pt)
practical_set2regex.eval()

special_symbols = "/.+*?^$()[]{}|\\"


def bf_escape(strings: list) -> list:
    processed_string = []
    for string in strings:
        temp: str = string
        for punct in p2s.keys():
            temp = temp.replace(punct, p2s[punct])
        processed_string.append(temp)
    return processed_string


class TimeOutException(Exception):
    pass


def is_last_sigma(lst: list, split_size):
    try:
        lst = lst[: lst.index(1)]
    except:
        pass
    split_size = vocab.stoi[str(split_size)]
    max_example_len = len(lst)
    try:
        idx = max_example_len - 1 - list(reversed(lst)).index(split_size)
    except:
        return False
    if idx != max_example_len - 1:
        return True
    else:
        return False


def sub_bf_esacpe(x):
    x = vocab.itos[x]
    if x in p2s.keys():
        x = p2s[x]
    return x


def split(strings, label, no_split=False, need_bf_escape=False):
    vocab = Vocabulary()
    splited_string = []
    # 그냥 padding이나 unk만 없애줌
    if no_split:
        # strings.max라고 하면 64(pad) 아니면 65(unk)인데 pad라고 보고 이를 처리하는 것이라고 명시해주는 것이 좋을듯
        for batch_idx in range(len(strings)):
            set = []
            for set_idx in range(10):
                seq = []
                if need_bf_escape:
                    seq.append(
                        "".join(
                            map(
                                sub_bf_esacpe,
                                [x for x in strings[batch_idx, set_idx] if x != vocab.stoi["<pad>"]],
                            )
                        )
                    )
                else:
                    seq.append(
                        "".join(
                            map(
                                lambda x: vocab.itos[x],
                                [x for x in strings[batch_idx, set_idx] if x != vocab.stoi["<pad>"]],
                            )
                        )
                    )
                set.append(seq)
            splited_string.append(set)
        return splited_string, None

    label = [i.tolist() for i in label]
    tmp = torch.LongTensor(label).transpose(0, 1).squeeze(-1).tolist()
    # tmp: n_examples * example_max_len

    split_size = torch.tensor(label)[torch.tensor(label) != vocab.stoi["<pad>"]].max().item()
    split_size = vocab.itos[split_size]
    split_size = int(split_size, 32)

    if any(map(lambda x: is_last_sigma(x, split_size), tmp)):
        split_size += 1

    label2 = []
    sigma_lst = []
    # each example
    for templete in tmp:
        tmp2 = []
        sigma_lst2 = []
        now = 0
        # each index
        for element in templete:
            if element == vocab.stoi["<pad>"]:
                break
            # Not sigma
            if element != vocab.stoi["0"]:
                element = vocab.itos[element]
                element = int(element, 32)
                if now != element:
                    for _ in range(element - len(sigma_lst2)):
                        sigma_lst2.append(False)
                tmp2.append(element)
                now = element
            else:
                if not sigma_lst2 or not sigma_lst2[-1]:
                    # sigma_lst2 has same dimension as split_size
                    sigma_lst2.append(True)
                # tmp2: subregex index
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
                    for _ in range(predict_seq_dict[seq_id]):
                        if need_bf_escape:
                            tmp += sub_bf_esacpe(src_seq[idx])
                        else:
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
def generate_regex_with_split_ar(sigma_lst, sub_id, sub_pos_set, sub_neg_set, split_model, count_limit, prefix, alphabet_size, data_type, return_dict):
    # Singleton
    if len(sub_pos_set) == 1:
        character = sub_pos_set.pop()
        if character and character in special_symbols:
            character = "\\" + character
        return_dict[sub_id] = character
        return

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


def generate_regex_with_split_bf(sub_id, sub_pos_set, sub_neg_set, split_model, count_limit, prefix, alphabet_size, return_dict):
    # sub_pos_set = set(bf_escape(sub_pos_set))
    # sub_neg_set = bf_escape(sub_neg_set)

    if not sub_pos_set:
        return_dict[sub_id] = "@epsilon"
        return

    # Singleton
    if len(sub_pos_set) == 1:
        character = sub_pos_set.pop()
        # No need to escape since we already map it to hex symbols
        return_dict[sub_id] = character
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
    class Ex:
        def __init__(self, pos, neg):
            self.pos = pos
            self.neg = neg

        def __str__(self):
            print(self.pos, self.neg)

    if len(sub_pos_set) == 1:
        character = sub_pos_set.pop()
        if character and character in special_symbols:
            character = "\\" + character
        return_dict[sub_id] = character
        return

    if sigma_lst is not None and any(list(map(lambda x: x[sub_id], sigma_lst))):
        tmp = get_sigma(Examples(pos=sub_pos_set, neg=sub_neg_set))
        return_dict[sub_id] = tmp
        return
    else:
        sub_pos_set = set(filter(None, sub_pos_set))
        sub_neg_set = set(filter(None, sub_neg_set))
        n = randint(0, 999999999)
        tmp = execute([Ex(list(sub_pos_set), list(sub_neg_set))], input_path=f"./RAM/inputs/{sub_id}/{n}", output_path=f"./RAM/outputs/{sub_id}/{n}")

    tmp = str(tmp).replace("++", "+").replace("?+", "+")

    return_dict[sub_id] = tmp


def generate_regex_with_split_sr(sigma_lst, sub_id, sub_pos_set, sub_neg_set, return_dict, data_type):
    def translate_examples(examples, data_type):
        example_max_len = 10 if data_type == "random" else 15
        translated_examples = []
        for example in examples:
            if example == "":
                translated_example = ["<pad>"] * example_max_len
            else:
                translated_example = list(example) + ["<pad>"] * (example_max_len - len(example))
            translated_examples.append(vocab.lookup_indices(translated_example[:example_max_len]))
        return torch.tensor(translated_examples)

    if len(sub_pos_set) == 1:
        character = sub_pos_set.pop()
        if character and character in special_symbols:
            character = "\\" + character
        return_dict[sub_id] = character
        return

    if sigma_lst is not None and any(list(map(lambda x: x[sub_id], sigma_lst))):
        tmp = get_sigma(Examples(pos=sub_pos_set, neg=sub_neg_set))
        return_dict[sub_id] = tmp
        return

    sub_pos_set = list(sub_pos_set) + [""] * (10 - len(sub_pos_set))
    sub_pos_set = translate_examples(sub_pos_set, data_type).unsqueeze(0)
    sub_neg_set = list(sub_neg_set) + [""] * (10 - len(sub_neg_set))
    sub_neg_set = translate_examples(sub_neg_set, data_type).unsqueeze(0)

    if data_type == "random":
        model = random_set2regex
    else:
        model = practical_set2regex

    _, _, other = model(sub_pos_set.cuda(), sub_neg_set.cuda())
    seqlist = other["sequence"]
    regex = torch.stack(seqlist, dim=0).squeeze(-1).squeeze(-1)
    regex = vocab.lookup_tokens(regex)
    regex = "".join([x for x in regex if x not in ("<sos>", "<eos>", "<pad>", "<unk>")])
    # print("regex:", regex)
    return_dict[sub_id] = regex


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
    is_split = type(splited_pos[0]) is list
    if is_split:
        split_size = len(splited_pos[0])
    else:
        split_size = 1

    # splited_pos = list(filter(lambda x: any(x), splited_pos))
    # splited_neg = list(filter(lambda x: any(x), splited_neg))

    split_set = []
    if is_split:
        if not splited_neg:
            splited_neg = [""]
        for sub_id in range(split_size):
            pos = []
            for set_idx in range(len(splited_pos)):
                pos.append(splited_pos[set_idx][sub_id])
            split_set.append([set(pos), set(splited_neg)])
    else:
        split_set.append([set(splited_pos), set(splited_neg)])

    # synthesis one by one
    for sub_id in range(split_size):
        # prefix strategy (only nth element or every element)
        # 마지막 원소거나 매번 프리픽스를 사용한다.
        if sub_id != 0 and (sub_id == split_size - 1 or use_prefix_every):
            # 전 prefix들을 ()로 감싼다.
            prefix = "(" + ")(".join([return_dict[i] for i in range(sub_id) if return_dict[i] != ""]) + ")"
        else:
            # neg와 pos가 겹치면 제외한다.
            split_set[sub_id][1] -= split_set[sub_id][0]
            prefix = None

        # print("Splited Positive Strings:", split_set[sub_id][0])
        # print("Splited Negative Strings:", split_set[sub_id][1])

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
            generate_regex_with_split_rg(sigma_lst, sub_id, split_set[sub_id][0], split_set[sub_id][1], return_dict)
        elif submodel == "set2regex":
            generate_regex_with_split_sr(sigma_lst, sub_id, split_set[sub_id][0], split_set[sub_id][1], return_dict, data_type)
        else:
            raise Exception("unknown baseline")

    return "(" + ")(".join([return_dict[i] for i in range(split_size) if return_dict[i] != ""]) + ")", split_size


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
