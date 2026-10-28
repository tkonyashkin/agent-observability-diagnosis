"""Согласие двух разметчиков на слепых карточках.

Запуск из корня репозитория:
    python3 kt2/service_eval/scripts/agreement.py \
        kt2/service_eval/annotation/labels_A.csv \
        kt2/service_eval/annotation/labels_B.csv

Вход: CSV с колонками id, failure_type, root_step, status или JSONL с теми же полями.
Результат печатается и сохраняется в annotation/agreement.json.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

FIELDS = ["failure_type", "root_step", "status"]
CARDS = Path(__file__).resolve().parents[1] / "annotation/blind_cards.jsonl"
FAILURE_TYPES = {"retrieval", "prompt", "model", "tool", "none"}
TYPE_STEPS = {
    "retrieval": {"search_documents.rank"},
    "prompt": {"build_parse_prompt", "build_answer_prompt"},
    "model": {"parse_query", "compose_answer"},
    "tool": {"search_documents"},
    "none": {None},
}
STATUSES = {"confirmed", "probable", "unresolved"}


def load(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            if row.get("root_step") == "":
                row["root_step"] = None
    else:
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if any(not isinstance(row, dict) or not row.get("id") for row in rows):
        raise ValueError(f"{path}: у строки нет id")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: повторяются id")
    expected_ids = {json.loads(l)["id"] for l in CARDS.read_text(encoding="utf-8").splitlines() if l.strip()}
    if set(ids) != expected_ids:
        missing = sorted(expected_ids - set(ids))
        extra = sorted(set(ids) - expected_ids)
        raise ValueError(f"{path}: неполная выборка, пропущены {missing}, лишние {extra}")
    for row in rows:
        if any(field not in row for field in FIELDS):
            raise ValueError(f"{path}: у {row['id']} нет обязательного поля")
        if row["failure_type"] not in FAILURE_TYPES or row["status"] not in STATUSES:
            raise ValueError(f"{path}: у {row['id']} неизвестный вид или статус")
        if row["root_step"] not in TYPE_STEPS[row["failure_type"]]:
            raise ValueError(f"{path}: у {row['id']} корневой шаг не соответствует виду сбоя")
    return {r["id"]: r for r in rows}


def cohen_kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return po, (1.0 if pe == 1 else (po - pe) / (1 - pe))


def main(path_a, path_b):
    la, lb = load(path_a), load(path_b)
    ids = sorted(la)
    result = {"n": len(ids), "status_scope": "preliminary_kt2", "fields": {}, "disagreements": []}
    for f in FIELDS:
        a = [str(la[i].get(f)) for i in ids]
        b = [str(lb[i].get(f)) for i in ids]
        po, kappa = cohen_kappa(a, b)
        result["fields"][f] = {"agreement": round(po, 4), "cohen_kappa": round(kappa, 4)}
    joint = [i for i in ids if la[i].get("failure_type") == lb[i].get("failure_type")
             and la[i].get("root_step") == lb[i].get("root_step")]
    result["fields"]["type_and_step"] = {"agreement": round(len(joint) / len(ids), 4)}
    for i in ids:
        diff = {f: [la[i].get(f), lb[i].get(f)] for f in FIELDS if la[i].get(f) != lb[i].get(f)}
        if diff:
            result["disagreements"].append({"id": i, **diff})
    out = Path(__file__).resolve().parents[1] / "annotation/agreement.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
