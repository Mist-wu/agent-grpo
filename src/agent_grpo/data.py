"""Source adapters and deterministic disjoint train/dev/final test partitions."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
from .schema import Task, write_jsonl

SOURCES = {"gsm8k": ("openai/gsm8k", "main", "mit"),
           "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", "cc-by-sa-4.0"),
           "gsm-symbolic": ("apple/GSM-Symbolic", "main", "cc-by-nc-nd-4.0")}


def normalize_row(dataset, row, split, index, revision="main"):
    source_id, config, license_name = SOURCES[dataset]
    documents = []
    if dataset == "hotpotqa":
        context = row["context"]
        if isinstance(context, dict):
            documents = [{"title": title, "text": " ".join(sentences)} for title, sentences in zip(context["title"], context["sentences"])]
        else:
            documents = [{"title": item["title"], "text": " ".join(item["sentences"])} for item in context]
    row_id = str(row.get("id", index))
    if dataset == "gsm-symbolic":
        row_id += ":" + str(row["instance"])
    source = {"repository": source_id, "config": config, "revision": revision, "license": license_name,
              "original_split": split, "original_id": row.get("original_id", row_id)}
    if dataset == "gsm-symbolic":
        source.update(template_id=row["id"], instance=row["instance"])
    mapped_split = "test" if split in {"test", "validation"} else split
    return Task(f"{dataset}:{split}:{row_id}", dataset, row["question"], row["answer"], mapped_split, documents, source)


def split_train(tasks, train_count=3000, dev_count=100, seed=42):
    if len(tasks) < train_count + dev_count:
        raise ValueError("not enough distinct rows for requested train and dev counts")
    indices = list(range(len(tasks)))
    random.Random(seed).shuffle(indices)
    train, dev = [], []
    for index in indices[:train_count]:
        train.append(replace(tasks[index], split="train"))
    for index in indices[train_count:train_count + dev_count]:
        dev.append(replace(tasks[index], split="dev"))
    return train, dev


def prepare(out_dir, train_count=3000, dev_count=100, test_count=500, seed=42):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_train, all_dev, all_test = [], [], []
    provenance = {}
    for dataset, (repository, config, license_name) in SOURCES.items():
        revision = HfApi().dataset_info(repository).sha
        provenance[dataset] = {"repository": repository, "config": config, "revision": revision, "license": license_name}
        ds = load_dataset(repository, config, revision=revision)
        if dataset != "gsm-symbolic":
            train_tasks = [normalize_row(dataset, row, "train", i, revision) for i, row in enumerate(ds["train"])]
            train, dev = split_train(train_tasks, train_count, dev_count, seed)
            all_train.extend(train)
            all_dev.extend(dev)
        source_split = "validation" if dataset == "hotpotqa" else "test"
        test_tasks = [normalize_row(dataset, row, source_split, i, revision) for i, row in enumerate(ds[source_split])]
        if dataset == "gsm-symbolic":
            # Keep full released instances: estimate uncertainty across templates, not pseudo-independent variants.
            all_test.extend(test_tasks)
        else:
            random.Random(seed).shuffle(test_tasks)
            all_test.extend(test_tasks[:test_count])
    for split, tasks in [("train", all_train), ("dev", all_dev), ("test", all_test)]:
        write_jsonl(out / f"{split}.jsonl", tasks)
    provenance.update(seed=seed, counts={"train": len(all_train), "dev": len(all_dev), "test": len(all_test)})
    for split in ["train", "dev", "test"]:
        provenance[f"{split}_sha256"] = hashlib.sha256((out / f"{split}.jsonl").read_bytes()).hexdigest()
    (out / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance
