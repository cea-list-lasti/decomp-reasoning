import pandas as pd
import re
import string



def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall


def eval_acc(prediction, answer):
    matched = 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0



def get_accuracy_cwq_webqsp(csv_path):
    df = pd.read_csv(csv_path)

    # Ensure necessary columns exist
    required_columns = {"pred", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")

    # Lists to store metrics
    acc_list = []
    hit_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    bad_calls = []

    for pred, label in zip(df["pred"], df["label"]):
        # Ensure strings and handle missing values
        pred = str(pred).replace("|", "\n") if pd.notna(pred) else ""
        label = str(label) if pd.notna(label) else ""

        prediction_list = pred.split("\n")
        answer_list = label.split("|")

        # Compute F1, Precision, and Recall
        f1_score, precision_score, recall_score = eval_f1(prediction_list, answer_list)
        f1_list.append(f1_score)
        precision_list.append(precision_score)
        recall_list.append(recall_score)

        # Compute Accuracy & Hit
        prediction_str = " ".join(prediction_list)
        acc_list.append(eval_acc(prediction_str, answer_list))
        hit = eval_hit(prediction_str, answer_list)
        if hit == 0:
            bad_calls.append((prediction_str, answer_list))
        hit_list.append(hit)

    # Compute aggregate metrics
    acc = sum(acc_list) * 100 / len(acc_list)
    hit = sum(hit_list) * 100 / len(hit_list)
    f1 = sum(f1_list) * 100 / len(f1_list)
    precision = sum(precision_list) * 100 / len(precision_list)
    recall = sum(recall_list) * 100 / len(recall_list)

    # Print results
    print(f"Accuracy: {acc:.4f}")
    print(f"Hit: {hit:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    return hit, bad_calls


eval_funcs = {
    "webqsp": get_accuracy_cwq_webqsp,
    "cwq": get_accuracy_cwq_webqsp
}
