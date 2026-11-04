"""Explicit stage commands; no automatic cloud allocation or paid API requests."""
import argparse
import json
import os
from .data import prepare
from .report import summarize
from .tools import probe_python_sandbox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("prepare")
    data.add_argument("--out", default="data")
    data.add_argument("--train-count", type=int, default=3000)
    data.add_argument("--dev-count", type=int, default=100)
    data.add_argument("--test-count", type=int, default=500)
    data.add_argument("--seed", type=int, default=42)
    for name in ["collect", "sft", "train", "evaluate"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--data", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--seed", type=int, default=42)
        if name == "collect":
            p.add_argument("--target", type=int, default=500)
            p.add_argument("--attempts-per-question", type=int, default=2)
            p.add_argument("--batch-size", type=int, default=32)
        if name == "sft":
            p.add_argument("--epochs", type=int, default=1)
            p.add_argument("--resume")
        if name in {"train", "evaluate"}:
            p.add_argument("--adapter")
        if name == "train":
            p.add_argument("--resume")
            p.add_argument("--dev")
        if name == "evaluate":
            p.add_argument("--condition", choices=["base-direct", "base-tools", "e2-lora", "cold-sft", "grpo"], required=True)
            p.add_argument("--batch-size", type=int, default=int(os.environ.get("E3_TOOL_BATCH_SIZE", "32")), help="Batch size for greedy evaluation")
    report = sub.add_parser("report")
    report.add_argument("results", nargs="+")
    report.add_argument("--out", default="results/formal-summary.json")
    rescore = sub.add_parser("rescore", help="Apply one scoring version uniformly to saved generations")
    rescore.add_argument("--data", required=True)
    rescore.add_argument("--input", required=True)
    rescore.add_argument("--out", required=True)
    sub.add_parser("probe-python")
    copy = sub.add_parser("copy-checkpoint")
    copy.add_argument("directory")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.out, args.train_count, args.dev_count, args.test_count, args.seed), indent=2))
    elif args.command == "collect":
        from .stages import collect
        collect(args.config, args.data, args.out, args.seed, args.target, args.attempts_per_question, args.batch_size)
    elif args.command == "sft":
        from .stages import sft
        sft(args.config, args.data, args.out, args.seed, args.epochs, args.resume)
    elif args.command == "train":
        from .training import training
        training(args.config, args.data, args.out, args.seed, args.adapter, args.resume, args.dev)
    elif args.command == "evaluate":
        from .stages import evaluate
        evaluate(args.config, args.data, args.out, args.condition, args.seed, args.adapter, args.batch_size)
    elif args.command == "report":
        summarize(args.results, args.out)
    elif args.command == "rescore":
        from .report import rescore_saved
        print(json.dumps(rescore_saved(args.data, args.input, args.out)))
    elif args.command == "copy-checkpoint":
        from .training import copy_checkpoint
        copy_checkpoint(args.directory)
    else:
        if not probe_python_sandbox():
            raise SystemExit("sandbox probe did not pass")
        print("Python tool isolation passed")


if __name__ == "__main__":
    main()
