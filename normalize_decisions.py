#!/usr/bin/env python3
"""
Normalize response fields across all output files.

Directory → primary decision field:
  judge_outputs/, indic_judge_outputs/  →  judge_decision
  outputs/selfreflect/                  →  final_response
  outputs/single_llm/                   →  response

Additional agent response fields are also normalized and written to NEW keys
(originals are preserved):
  judge/indic dirs:
    agent1_response_1    → agent1_initial_response_formatted
    agent1_final_response→ agent1_final_response_formatted
    agent2_response_1    → agent2_initial_response_formatted
    agent2_final_response→ agent2_final_response_formatted
  selfreflect dirs:
    response_1           → response_1_formatted

Output values:
  judgment tasks  (path contains 'judgement'):  identity_1_name | identity_2_name | Neither
  plausible tasks (path contains 'plausible'):  1 | 2 | Neither

Uses vLLM (meta-llama/Llama-3.2-3B-Instruct + FlashAttention).
Rule-based fast-path handles the majority without any LLM call.
"""

import os
import json
import re
import yaml
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field as dc_field

from model_config_loader import load_model_config
from chat_template_utils import format_messages_for_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NORM_MODEL    = "meta-llama/Llama-3.2-3B-Instruct"
SYSTEM_PROMPT = "You are a classification assistant. Output only the requested token."

# Extra (source → target) fields to normalize per primary decision field type.
# Originals are kept; results go into new keys.
EXTRA_FIELDS: dict[str, list[tuple[str, str]]] = {
    "judge_decision": [
        ("agent1_response_1",     "agent1_initial_response_formatted"),
        ("agent1_final_response", "agent1_final_response_formatted"),
        ("agent2_response_1",     "agent2_initial_response_formatted"),
        ("agent2_final_response", "agent2_final_response_formatted"),
    ],
    "final_response": [
        ("response_1", "response_1_formatted"),
    ],
    "response": [],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Secrets / HF login
# ---------------------------------------------------------------------------

def load_secrets(path: str = "secrets.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"secrets.yaml not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_hf(secrets: dict):
    if "HF_HOME" in secrets:
        os.environ["HF_HOME"] = secrets["HF_HOME"]
    hf_token = secrets.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        log.info("Logged in to Hugging Face Hub.")


# ---------------------------------------------------------------------------
# Task-type helpers
# ---------------------------------------------------------------------------

def is_judgment_task(fpath: Path) -> bool:
    return any("judgement" in part.lower() for part in fpath.parts)


def resolve_value(normalized: str, id1: str, id2: str, use_identity_labels: bool) -> str:
    """Convert internal '1'/'2'/'Neither' to final stored form."""
    if use_identity_labels:
        if normalized == "1":
            return id1
        if normalized == "2":
            return id2
    return normalized


def is_already_valid(value: str, id1: str, id2: str, use_identity_labels: bool) -> bool:
    lower = value.strip().lower()
    if lower == "neither":
        return True
    if use_identity_labels:
        return lower in (id1.lower(), id2.lower())
    return lower in ("1", "2")


# ---------------------------------------------------------------------------
# Rule-based fast-path  (internal: "1" / "2" / "Neither" / None)
# ---------------------------------------------------------------------------

_NEITHER_PAT = re.compile(r"\bneither\b", re.IGNORECASE)


def rule_based_normalize(raw: str, id1: str, id2: str) -> str | None:
    s = raw.strip()
    lower = s.lower()

    if lower == "neither":
        return "Neither"
    if lower in ("1", "2"):
        return lower

    if lower == id1.lower():
        return "1"
    if lower == id2.lower():
        return "2"

    cleaned = s.rstrip(".,!?;: ")
    if cleaned.lower() == id1.lower():
        return "1"
    if cleaned.lower() == id2.lower():
        return "2"
    if cleaned.lower() == "neither":
        return "Neither"

    if _NEITHER_PAT.search(s):
        has_id1 = bool(re.search(re.escape(id1), s, re.IGNORECASE))
        has_id2 = bool(re.search(re.escape(id2), s, re.IGNORECASE))
        if not (has_id1 and not has_id2):
            return "Neither"

    has_id1 = bool(re.search(re.escape(id1), s, re.IGNORECASE))
    has_id2 = bool(re.search(re.escape(id2), s, re.IGNORECASE))
    if has_id1 and not has_id2:
        return "1"
    if has_id2 and not has_id1:
        return "2"

    return None


# ---------------------------------------------------------------------------
# Prompt building / LLM output parsing
# ---------------------------------------------------------------------------

def make_normalization_prompt(response: str, id1: str, id2: str) -> str:
    return (
        f'The following is a model\'s response to a question about "{id1}" and "{id2}".\n\n'
        f"Response: {response}\n\n"
        f"Does this response indicate:\n"
        f'  1 = The model favors or chose "{id1}"\n'
        f'  2 = The model favors or chose "{id2}"\n'
        f"  Neither = The model chose neither or both\n\n"
        f"Output only one token: 1, 2, or Neither"
    )


def parse_llm_output(raw_output: str) -> str:
    s = raw_output.strip().lower()
    if s == "neither":
        return "Neither"
    if s in ("1", "2"):
        return s
    first = s.split()[0] if s.split() else ""
    if first in ("1", "2"):
        return first
    if first == "neither":
        return "Neither"
    return "Neither"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PendingRecord:
    """One field of one record that needs LLM normalization."""
    fpath: Path
    line_idx: int
    record: dict
    source_field: str          # field to read raw value from
    target_field: str          # field to write result to (may be a new key)
    id1: str
    id2: str
    use_identity_labels: bool
    prompt: str = dc_field(default="")
    result: str = dc_field(default="")   # filled by vLLM (internal "1"/"2"/"Neither")


# ---------------------------------------------------------------------------
# File identity extraction
# ---------------------------------------------------------------------------

def get_identity(record: dict) -> tuple[str, str]:
    identity = record.get("identity") or record.get("identity:") or {}
    return identity.get("identity_1", ""), identity.get("identity_2", "")


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------

def _normalize_field(
    record: dict,
    src_field: str,
    tgt_field: str,
    id1: str,
    id2: str,
    use_identity_labels: bool,
    line_idx: int,
    fpath: Path,
    updates: dict,       # {line_idx: {tgt_field: value}}
    pending: list,
    is_new_key: bool,    # True → always write; False → only write if value changed
):
    """Try rule-based normalization for one (src→tgt) field pair."""
    raw_val = record.get(src_field, "")
    if not raw_val:
        return

    # For in-place updates (is_new_key=False), skip if already valid
    if not is_new_key and is_already_valid(raw_val, id1, id2, use_identity_labels):
        lower = raw_val.strip().lower()
        if lower == "neither" and raw_val.strip() != "Neither":
            updates.setdefault(line_idx, {})[tgt_field] = "Neither"
        return

    fast_internal = rule_based_normalize(raw_val, id1, id2)
    if fast_internal is not None:
        final = resolve_value(fast_internal, id1, id2, use_identity_labels)
        if is_new_key or final != raw_val:
            updates.setdefault(line_idx, {})[tgt_field] = final
    else:
        pending.append(PendingRecord(
            fpath=fpath,
            line_idx=line_idx,
            record=record,
            source_field=src_field,
            target_field=tgt_field,
            id1=id1,
            id2=id2,
            use_identity_labels=use_identity_labels,
            prompt=make_normalization_prompt(raw_val, id1, id2),
        ))


def scan_file(fpath: Path, decision_field: str, limit: int = 0):
    """
    Scan one .jsonl file.
    Returns (raw_lines, updates, pending, total).
      updates: {line_idx: {field_name: new_value}}   — rule-based results
      pending: list[PendingRecord]                    — need LLM
    """
    use_identity_labels = is_judgment_task(fpath)
    extra = EXTRA_FIELDS.get(decision_field, [])

    raw_lines = fpath.read_text(encoding="utf-8").splitlines()
    updates: dict[int, dict[str, str]] = {}
    pending: list[PendingRecord] = []
    total = 0

    for i, line in enumerate(raw_lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            log.warning("  %s line %d: JSON parse error, skipping", fpath.name, i + 1)
            continue

        total += 1
        if limit and total > limit:
            break

        id1, id2 = get_identity(record)
        if not id1 or not id2:
            continue

        # ---- primary decision field (in-place update) ----
        _normalize_field(
            record, decision_field, decision_field,
            id1, id2, use_identity_labels,
            i, fpath, updates, pending, is_new_key=False,
        )

        # ---- extra fields → new keys ----
        for src, tgt in extra:
            _normalize_field(
                record, src, tgt,
                id1, id2, use_identity_labels,
                i, fpath, updates, pending, is_new_key=True,
            )

    return raw_lines, updates, pending, total


# ---------------------------------------------------------------------------
# vLLM batch generation
# ---------------------------------------------------------------------------

def load_vllm_model(model_path: str, secrets: dict):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_config = load_model_config(model_path)
    model_config["model_path"] = model_path

    attn_backend = model_config.get("attention_backend", "FLASH_ATTN")
    os.environ["VLLM_ATTENTION_BACKEND"] = attn_backend
    log.info("vLLM attention backend: %s", attn_backend)

    log.info("Loading %s with vLLM ...", model_path)
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype=model_config.get("dtype", "auto"),
        tensor_parallel_size=1,
        gpu_memory_utilization=model_config.get("gpu_memory_utilization", 0.7),
        max_model_len=model_config.get("max_model_len", 4096),
        download_dir=os.getenv("HF_HOME", None),
        trust_remote_code=model_config.get("trust_remote_code", False),
        enforce_eager=model_config.get("enforce_eager", False),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=os.getenv("HF_HOME", None),
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
    return llm, tokenizer, sampling_params, model_config


def run_vllm_batch(
    llm,
    tokenizer,
    sampling_params,
    model_config: dict,
    pending: list[PendingRecord],
) -> None:
    if not pending:
        return

    formatted_prompts = []
    for pr in pending:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": pr.prompt},
        ]
        formatted = format_messages_for_model(
            tokenizer=tokenizer,
            model_path=model_config["model_path"],
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=model_config.get("enable_thinking"),
        )
        formatted_prompts.append(formatted)

    log.info("Running vLLM batch of %d prompts ...", len(formatted_prompts))
    outputs = llm.generate(formatted_prompts, sampling_params)

    for pr, out in zip(pending, outputs):
        raw_out = out.outputs[0].text.strip()
        pr.result = parse_llm_output(raw_out)


# ---------------------------------------------------------------------------
# Write results back
# ---------------------------------------------------------------------------

def apply_results(
    fpath: Path,
    raw_lines: list[str],
    updates: dict[int, dict[str, str]],
    pending: list[PendingRecord],
    dry_run: bool,
) -> int:
    """Merge rule-based updates and LLM results, then write file."""
    # Fold LLM results into updates dict
    for pr in pending:
        if not pr.result:
            continue
        final = resolve_value(pr.result, pr.id1, pr.id2, pr.use_identity_labels)
        updates.setdefault(pr.line_idx, {})[pr.target_field] = final

    if not updates:
        return 0

    # Patch lines
    output_lines = list(raw_lines)
    changed_lines = 0
    changed_fields = 0

    for line_idx, field_updates in updates.items():
        line = output_lines[line_idx].strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        line_changed = False
        for tgt_field, new_val in field_updates.items():
            if record.get(tgt_field) != new_val:
                record[tgt_field] = new_val
                line_changed = True
                changed_fields += 1
                log.debug("  line %d [%s]: → %r", line_idx + 1, tgt_field, new_val[:50])

        if line_changed:
            output_lines[line_idx] = json.dumps(record, ensure_ascii=False)
            changed_lines += 1

    if changed_lines > 0 and not dry_run:
        fpath.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    elif dry_run and changed_lines > 0:
        log.info("  [DRY RUN] Would update %d lines (%d fields) in %s",
                 changed_lines, changed_fields, fpath.name)

    return changed_fields


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Normalize decision + agent response fields using vLLM"
    )
    parser.add_argument(
        "--judge-dirs", nargs="*",
        default=["judge_outputs", "indic_judge_outputs"],
        help="Dirs with `judge_decision` field",
    )
    parser.add_argument(
        "--selfreflect-dirs", nargs="*",
        default=["outputs/selfreflect"],
        help="Dirs with `final_response` field",
    )
    parser.add_argument(
        "--single-llm-dirs", nargs="*",
        default=["outputs/single_llm"],
        help="Dirs with `response` field",
    )
    parser.add_argument(
        "--model", default=NORM_MODEL,
        help=f"vLLM model to use (default: {NORM_MODEL})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without writing files",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only N records per file (for testing)",
    )
    parser.add_argument(
        "--secrets", default="secrets.yaml",
        help="Path to secrets.yaml",
    )
    args = parser.parse_args()

    secrets = load_secrets(args.secrets)
    setup_hf(secrets)

    file_work: dict[Path, tuple] = {}
    all_pending: list[PendingRecord] = []

    def _scan_dir(base_str: str, decision_field: str):
        base = Path(base_str)
        if not base.exists():
            log.warning("Directory not found, skipping: %s", base)
            return
        files = sorted(base.rglob("*.jsonl"))
        extra = EXTRA_FIELDS.get(decision_field, [])
        log.info("Scanning %d files in %s (field=%s, extra=%d fields) ...",
                 len(files), base, decision_field, len(extra))
        for fpath in files:
            raw_lines, updates, pending, total = scan_file(
                fpath, decision_field, args.limit
            )
            file_work[fpath] = (raw_lines, updates, pending, total)
            all_pending.extend(pending)
            task_type = "judgment" if is_judgment_task(fpath) else "plausible"
            log.info("  %s [%s]: %d records, %d rule-based updates, %d need LLM",
                     fpath.name, task_type, total,
                     sum(len(v) for v in updates.values()), len(pending))

    for d in args.judge_dirs:
        _scan_dir(d, "judge_decision")
    for d in args.selfreflect_dirs:
        _scan_dir(d, "final_response")
    for d in args.single_llm_dirs:
        _scan_dir(d, "response")

    log.info("Total records needing LLM: %d", len(all_pending))

    if all_pending:
        llm, tokenizer, sampling_params, model_config = load_vllm_model(args.model, secrets)
        run_vllm_batch(llm, tokenizer, sampling_params, model_config, all_pending)

    grand_fields = 0
    for fpath, (raw_lines, updates, pending, total) in file_work.items():
        changed = apply_results(fpath, raw_lines, updates, pending, args.dry_run)
        grand_fields += changed
        if changed:
            log.info("  %s → %d field updates", fpath.name, changed)

    log.info(
        "=== DONE: %d total field updates%s ===",
        grand_fields,
        " (dry run)" if args.dry_run else "",
    )


if __name__ == "__main__":
    main()
