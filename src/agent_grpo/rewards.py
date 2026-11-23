"""Answer scoring, with explicit numeric extraction and HotpotQA normalization."""
from collections import Counter
from decimal import Decimal, InvalidOperation
import re
import string

SCORE_VERSION = "v2-qwen-gsm-last-number-hotpot-f1"
LAST_NUMBER = re.compile(r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)")
FINAL_TAG = re.compile(r"<final>[^<>]+</final>\s*$")


def is_final_format_compliant(text):
    return bool(FINAL_TAG.search(text))


def final_answer(text):
    if "<final>" in text:
        match = re.search(r"<final>(.*?)</final>", text, re.S)
        return match.group(1).strip() if match else ""
    if "####" in text:
        return text.rsplit("####", 1)[1].strip()
    return text.strip()


def numeric_answer(text):
    text = final_answer(text).replace(",", "").strip()
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1].strip()
    else:
        # Accept an explicit final answer/result clause, not an arbitrary last number
        # in working. This separates answer correctness from XML compliance.
        conclusion = re.search(r"(?:answer|result)[^\n]*?(?:is|=)\s*([-+]?\d+(?:\.\d+)?)(?:\s*[.!])?\s*$", text, re.I)
        if conclusion:
            text = conclusion.group(1)
    # A final field must contain exactly one numeric answer, not arbitrary working.
    text = text.removeprefix("$").removesuffix("$").strip()
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def numeric_answer_flexible(text):
    """Qwen's published GSM8K last-number extraction; no execution of model text."""
    matches = list(LAST_NUMBER.finditer(text))
    if not matches:
        return None
    return Decimal(matches[-1].group().replace(",", "").replace("+", "").strip())


def normalize_answer(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def hotpot_f1(prediction, answer):
    p, g = normalize_answer(final_answer(prediction)), normalize_answer(answer)
    if (p in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"}) and p != g:
        return 0.0
    p, g = p.split(), g.split()
    if not p or not g:
        return float(p == g)
    shared = sum((Counter(p) & Counter(g)).values())
    return 2 * shared / (len(p) + len(g))


def score_answer_strict(dataset, prediction, answer):
    if dataset == "hotpotqa":
        return hotpot_f1(prediction, answer)
    pred, gold = numeric_answer(prediction), numeric_answer(answer)
    return float(pred is not None and gold is not None and pred == gold)


def score_answer(dataset, prediction, answer):
    if dataset == "hotpotqa":
        return hotpot_f1(prediction, answer)
    pred, gold = numeric_answer_flexible(prediction), numeric_answer_flexible(answer)
    return float(pred is not None and gold is not None and pred == gold)


def reward(task, text, calls, parsed_calls):
    correctness = score_answer(task.dataset, text, task.answer)
    # One per-trajectory parsing bonus, never a bonus per repeated tool call.
    parse_bonus = 0.1 if parsed_calls > 0 else 0.0
    tool_penalty = 0.05 * calls
    return {"total": correctness + parse_bonus - tool_penalty,
            "correctness": correctness, "parse_bonus": parse_bonus,
            "tool_penalty": tool_penalty, "tool_calls": calls,
            "strict_correctness_v1": score_answer_strict(task.dataset, text, task.answer),
            "final_format_compliant": is_final_format_compliant(text),
            "score_version": SCORE_VERSION}
