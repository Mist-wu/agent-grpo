"""JSONL experiment input and trajectory records."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


@dataclass
class Task:
    id: str
    dataset: str
    question: str
    answer: str
    split: str
    documents: list[dict] = field(default_factory=list)
    source: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test")
        if not self.id or not self.question or not self.answer:
            raise ValueError("id/question/answer must be nonempty")


@dataclass
class Trajectory:
    task_id: str
    token_ids: list[int]
    generated_mask: list[bool]
    final_text: str
    turns: list[dict]
    reward: dict
    truncated: bool = False

    def __post_init__(self):
        if len(self.token_ids) != len(self.generated_mask):
            raise ValueError("token and mask lengths differ")
        if self.generated_mask and self.generated_mask[0]:
            raise ValueError("first token cannot be a predicted action")


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_tasks(path):
    tasks = [Task(**row) for row in read_jsonl(path)]
    ids = [task.id for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task IDs")
    return tasks


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(asdict(row) if hasattr(row, "__dataclass_fields__") else row, ensure_ascii=False, allow_nan=False) + "\n")


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
