"""Проверка eval-сета сервиса диагностики.

Запуск из корня репозитория (Python 3.10+, только стандартная библиотека):
    python3 kt2/service_eval/scripts/validate.py

Проверяет схему кейсов, состав классов, связь с кейсами Junkie и отсутствие
пересечений dev/holdout по group_id. По умолчанию проверяет только открытый
манифест holdout. Содержимое проверяется только с --holdout ПУТЬ.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KT2 = ROOT / "kt2"
SERVICE = KT2 / "service_eval"

STEPS = ["build_parse_prompt", "parse_query", "search_documents",
         "search_documents.rank", "build_answer_prompt", "compose_answer"]
VERDICTS = {"ok", "fail", "absent"}
STATUSES = {"confirmed", "probable", "unresolved"}
TYPE_STEPS = {
    "retrieval": {"search_documents.rank"},
    "tool": {"search_documents"},
    "prompt": {"build_parse_prompt", "build_answer_prompt"},
    "model": {"parse_query", "compose_answer"},
}
EXPECTED_COUNTS = {
    "dev": {"retrieval": 11, "prompt": 11, "model": 11, "tool": 11,
            "no_failure": 4, "incomplete": 4, "compound": 2},
    "holdout": {"retrieval": 5, "prompt": 5, "model": 5, "tool": 5,
                "no_failure": 3, "incomplete": 2, "compound": 1},
}
# Кейсы Junkie, из которых берутся примеры для промпта диагноста. В eval не входят.
FEW_SHOT_RESERVE = {"E03", "E24", "E45", "E97"}

errors = []

parser = argparse.ArgumentParser(description="Проверка eval-сета сервиса")
parser.add_argument("--holdout", type=Path, help="явно проверить закрытый набор и его хеш")
args = parser.parse_args()


def err(msg):
    errors.append(msg)


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


junkie = {r["id"]: r for r in load_jsonl(KT2 / "agent_eval/dev.jsonl")}
final_gold = {r["id"]: r for r in load_jsonl(KT2 / "agent_eval/final_gold.jsonl")}
REFERENCE_FIELDS = ("action", "relevant_doc_ids", "filters", "order_by", "has_more")
for jid, gold in final_gold.items():
    if jid not in junkie:
        err(f"Junkie {jid}: нет в dev.jsonl")
        continue
    for key in REFERENCE_FIELDS:
        if junkie[jid]["expected"][key] != gold[key]:
            err(f"Junkie {jid}: {key} в dev.jsonl не совпадает с final_gold.jsonl")
doc_ids = {d["doc_id"] for d in load_jsonl(KT2 / "agent_eval/documents.jsonl")}
env = json.loads((KT2 / "agent_eval/fixtures/environment.json").read_text(encoding="utf-8"))
for fx in env["fixtures"].values():
    for d in fx.get("additional_documents", []):
        doc_ids.add(d["doc_id"])

reserve_groups = {junkie[i]["group_id"] for i in FEW_SHOT_RESERVE}


def check_case(c, split):
    cid = c["id"]
    if c["split"] != split:
        err(f"{cid}: split={c['split']}, ожидался {split}")
    a = c["agent_case"]
    j = junkie.get(a["junkie_id"])
    if j is None:
        err(f"{cid}: нет кейса Junkie {a['junkie_id']}")
        return
    for k_case, k_j in (("query", "query"), ("fixture_id", "fixture_id"),
                        ("user_id", "user_id"), ("junkie_group_id", "group_id")):
        if a[k_case] != j[k_j]:
            err(f"{cid}: {k_case} не совпадает с Junkie {a['junkie_id']}")
    if a["junkie_id"] in FEW_SHOT_RESERVE or a["junkie_group_id"] in reserve_groups:
        err(f"{cid}: использует кейс из резерва few-shot")
    for key in REFERENCE_FIELDS:
        if c["agent_reference"][key] != j["expected"][key]:
            err(f"{cid}: agent_reference.{key} не совпадает с Junkie {a['junkie_id']}")
    for d in c["agent_reference"]["relevant_doc_ids"]:
        if d not in doc_ids:
            err(f"{cid}: неизвестный doc_id {d}")

    e = c["expected"]
    v = e["step_verdicts"]
    if list(v) != STEPS:
        err(f"{cid}: шаги {list(v)} не совпадают со списком STEPS")
    if set(v.values()) - VERDICTS:
        err(f"{cid}: недопустимые вердикты {set(v.values()) - VERDICTS}")
    if e["status"] not in STATUSES:
        err(f"{cid}: недопустимый статус {e['status']}")
    if e["confirmable"] != (e["status"] == "confirmed"):
        err(f"{cid}: confirmable не согласован со статусом")
    for s in e["unobservable_steps"]:
        if s not in STEPS:
            err(f"{cid}: неизвестный шаг {s} в unobservable_steps")

    cls = c["failure_class"]
    if cls == "no_failure":
        if e["failure_type"] != "none" or e["root_step"] is not None or "fail" in v.values():
            err(f"{cid}: контрольный кейс должен быть без сбоя")
        if c["injection"] is not None:
            err(f"{cid}: в контрольном кейсе есть поломка")
        return
    root = e["root_step"]
    if v.get(root) != "fail":
        err(f"{cid}: корневой шаг {root} не помечен fail")
    if root not in TYPE_STEPS.get(e["failure_type"], set()):
        err(f"{cid}: шаг {root} не относится к виду {e['failure_type']}")
    if cls in TYPE_STEPS and e["failure_type"] != cls:
        err(f"{cid}: класс {cls}, а вид {e['failure_type']}")
    if cls == "compound" and not e["secondary"]:
        err(f"{cid}: в составном кейсе нет второй причины")
    for sec in e["secondary"]:
        if sec["step"] not in TYPE_STEPS[sec["failure_type"]] or v.get(sec["step"]) != "fail":
            err(f"{cid}: вторая причина {sec} не согласована с вердиктами")
    if cls == "incomplete":
        tv = c["trace_variant"]
        if all(tv.values()):
            err(f"{cid}: в кейсе неполного трейса все данные доступны")
        if e["status"] == "confirmed":
            err(f"{cid}: неполный трейс не может иметь статус confirmed")


def check_split(rows, split):
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        err(f"{split}: повторяющиеся id")
    counts = Counter(r["failure_class"] for r in rows)
    if dict(counts) != EXPECTED_COUNTS[split]:
        err(f"{split}: состав {dict(counts)} не совпадает с {EXPECTED_COUNTS[split]}")
    for r in rows:
        check_case(r, split)
    return counts


dev = load_jsonl(SERVICE / "dev.jsonl")
dev_counts = check_split(dev, "dev")
dev_groups = {r["agent_case"]["junkie_group_id"] for r in dev}

manifest = json.loads((SERVICE / "holdout/manifest.json").read_text(encoding="utf-8"))
if args.holdout is not None:
    if not args.holdout.is_file():
        parser.error(f"файл holdout не найден: {args.holdout}")
    holdout = load_jsonl(args.holdout)
    check_split(holdout, "holdout")
    if sha256(args.holdout) != manifest["sha256_plaintext"]:
        err("holdout: хеш не совпадает с манифестом (файл изменён после заморозки)")
    holdout_groups = {r["agent_case"]["junkie_group_id"] for r in holdout}
    if set(manifest["junkie_group_ids"]) != holdout_groups:
        err("holdout: group_id в манифесте не совпадают с файлом")
    state = "проверен локально"
else:
    holdout_groups = set(manifest["junkie_group_ids"])
    state = "скрыт, содержимое и хеш не проверены"

if dev_counts and dict(Counter(manifest["counts"])) != EXPECTED_COUNTS["holdout"]:
    err("holdout: состав в манифесте не совпадает с ожидаемым")
overlap = dev_groups & holdout_groups
if overlap:
    err(f"dev и holdout пересекаются по group_id Junkie: {sorted(overlap)}")
if reserve_groups & holdout_groups:
    err("holdout использует группы резерва few-shot")

print(f"dev: {len(dev)} кейсов, {dict(dev_counts)}")
print(f"holdout: {sum(manifest['counts'].values())} кейсов, {state}")
print(f"dev sha256: {sha256(SERVICE / 'dev.jsonl')}")
if errors:
    print(f"\nОшибок: {len(errors)}")
    for e in errors:
        print(" -", e)
    sys.exit(1)
print("OK")
