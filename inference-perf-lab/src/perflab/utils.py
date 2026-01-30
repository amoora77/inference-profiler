import json
import os
from datetime import datetime
from pathlib import Path


def mkdirp(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def timestamp_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def append_jsonl(path, record):
    mkdirp(os.path.dirname(path))
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
