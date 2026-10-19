# -*- coding: utf-8 -*-
"""
Judge evaluation script using GPT-5-Nano via OpenAI batch API.

Processes debate results from multi_agent_same or multi_agent_diff outputs
and generates judge's final decisions using gpt-5-nano-2025-08-07.
"""

import os
import json
import time
import yaml
import logging
import argparse
from datetime import datetime
from openai import OpenAI
import judge_prompts as judge_module

MODEL = "gpt-5-nano-2025-08-07"
SYSTEM_PROMPT = "You are a rigorous evaluation assistant. Follow all instructions carefully."

# Setup logging
os.makedirs("logs", exist_ok=True)
now_str = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    filename=f"logs/judge_gpt_{now_str}.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_secrets(secrets_path: str = "secrets.yaml") -> dict:
    if not os.path.exists(secrets_path):
        raise FileNotFoundError(
            f"secrets.yaml not found at {secrets_path}. "
            "Please create it with an 'openai.api_key' entry."
        )
    with open(secrets_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_input_files(input_dirs: list, output_base: str) -> list:
    """
    Walk each input directory recursively and return (input_path, output_path)
    pairs for every .jsonl file found. Already-processed files are skipped.
    """
    pairs = []
    for input_dir in input_dirs:
        input_dir = os.path.abspath(input_dir)
        for root, _, files in os.walk(input_dir):
            for fname in sorted(files):
                if not fname.endswith(".jsonl"):
                    continue
                input_path = os.path.join(root, fname)
                rel = os.path.relpath(input_path, os.path.dirname(input_dir))
                output_path = os.path.join(output_base, rel)
                if os.path.exists(output_path):
                    print(f"Skipping (already done): {input_path}")
                    continue
                pairs.append((input_path, output_path))
    return pairs


def build_batch_requests(debates: list, debug: bool = False) -> list:
    """Build OpenAI batch request dicts for each debate record."""
    template = judge_module.judge_prompts["judge_decision"]
    requests = []
    for debate in debates:
        p = template
        p = p.replace("{{prompt}}", debate.get("prompt", ""))
        p = p.replace("{{model1_response}}", debate.get("agent1_response_1", ""))
        p = p.replace("{{model2_response}}", debate.get("agent2_response_1", ""))
        p = p.replace("{{model1_feedback}}", debate.get("agent1_feedback", ""))
        p = p.replace("{{model2_feedback}}", debate.get("agent2_feedback", ""))
        p = p.replace("{{model1_final}}", debate.get("agent1_final_response", ""))
        p = p.replace("{{model2_final}}", debate.get("agent2_final_response", ""))

        requests.append({
            "custom_id": f"request_{debate['id']}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p}
                ]
            }
        })

    if debug:
        requests = requests[:10]

    return requests


def write_batch_jsonl(requests: list, batch_path: str):
    os.makedirs(os.path.dirname(batch_path) or ".", exist_ok=True)
    with open(batch_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    logging.info(f"Wrote {len(requests)} requests to {batch_path}")
    print(f"  Wrote {len(requests)} requests to {batch_path}")


def upload_file(client: OpenAI, batch_path: str) -> str:
    with open(batch_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    logging.info(f"Uploaded file: {uploaded.id}")
    print(f"  Uploaded file: {uploaded.id}")
    return uploaded.id


def submit_batch(client: OpenAI, file_id: str) -> object:
    job = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    logging.info(f"Batch job created: {job.id}")
    print(f"  Batch job created: {job.id}")
    return job


def poll_until_done(client: OpenAI, batch_id: str, interval: int = 30) -> object:
    terminal = {"completed", "failed", "cancelled", "expired"}
    while True:
        batch = client.batches.retrieve(batch_id)
        state = batch.status
        print(f"  Status: {state}  |  {batch.request_counts}")
        logging.info(f"{batch_id} -> {state} | {batch.request_counts}")
        if state in terminal:
            return batch
        time.sleep(interval)


def download_result(client: OpenAI, output_file_id: str, raw_output_path: str):
    content = client.files.content(output_file_id).read().decode("utf-8-sig")
    os.makedirs(os.path.dirname(raw_output_path) or ".", exist_ok=True)
    with open(raw_output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logging.info(f"Raw results saved to {raw_output_path}")
    print(f"  Raw results saved to: {raw_output_path}")


def parse_decisions(raw_path: str) -> dict:
    """Parse raw OpenAI batch results JSONL into a custom_id -> decision text map."""
    decisions = {}
    with open(raw_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            custom_id = record.get("custom_id", "")
            try:
                text = record["response"]["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                text = ""
            decisions[custom_id] = text.strip()
    return decisions


def save_results(debates: list, decisions: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for debate in debates:
            custom_id = f"request_{debate['id']}"
            debate["judge_decision"] = decisions.get(custom_id, "")
            debate["judge_model"] = MODEL
            f.write(json.dumps(debate, ensure_ascii=False) + "\n")
    print(f"  Saved {len(debates)} records to {output_path}")
    logging.info(f"Saved {len(debates)} records to {output_path}")


def process_file(client: OpenAI, input_path: str, output_path: str,
                 batch_dir: str, debug: bool):
    """Full pipeline for a single input file."""
    print(f"\n  Input : {input_path}")
    print(f"  Output: {output_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        debates = [json.loads(line) for line in f]
    print(f"  Loaded {len(debates)} debate records")

    requests = build_batch_requests(debates, debug=debug)

    stem = os.path.splitext(os.path.basename(input_path))[0]
    batch_path = os.path.join(batch_dir, f"{stem}_batch.jsonl")
    raw_path = os.path.join(batch_dir, f"{stem}_raw_results.jsonl")

    write_batch_jsonl(requests, batch_path)

    file_id = upload_file(client, batch_path)
    job = submit_batch(client, file_id)
    finished_job = poll_until_done(client, job.id)

    if finished_job.status != "completed":
        print(f"  Job did not complete: {finished_job.status}")
        logging.error(f"Job failed for {input_path}: {finished_job.status}")
        if finished_job.error_file_id:
            download_result(client, finished_job.error_file_id,
                            raw_path.replace(".jsonl", "_errors.jsonl"))
        return

    download_result(client, finished_job.output_file_id, raw_path)
    decisions = parse_decisions(raw_path)
    save_results(debates, decisions, output_path)


def main():
    parser = argparse.ArgumentParser(
        description=f"Judge evaluation using {MODEL} (OpenAI batch API)."
    )

    # --- mode flags ---
    parser.add_argument("--run", action="store_true",
                        help="Submit a batch job for a single input file (returns job ID).")
    parser.add_argument("--poll", action="store_true",
                        help="Poll a batch job until it reaches a terminal state.")
    parser.add_argument("--retrieve", action="store_true",
                        help="Download raw results of a completed batch job.")
    parser.add_argument("--merge", action="store_true",
                        help="Merge downloaded raw results back into debate records.")
    parser.add_argument("--cancel", action="store_true",
                        help="Cancel an ongoing batch job.")

    # --- per-action arguments ---
    parser.add_argument("--batch-job-id", type=str,
                        help="Batch job ID for --poll / --retrieve / --cancel.")
    parser.add_argument("--input-file", type=str,
                        help="Single input JSONL file for --run or --merge.")
    parser.add_argument("--batch-file", type=str,
                        help="Path to write the intermediate batch JSONL (used with --run).")
    parser.add_argument("--raw-results", type=str,
                        help="Path to raw batch results JSONL (used with --retrieve / --merge).")
    parser.add_argument("--output-file", type=str,
                        help="Output path for --merge.")

    # --- full-pipeline (default) arguments ---
    parser.add_argument("--input-dirs", nargs="+",
                        default=["outputs/multi_agent_diff", "outputs/multi_agent_same"],
                        help="Directories to scan recursively (default full-pipeline mode).")
    parser.add_argument("--output-dir", default="judge_outputs/gpt_nano",
                        help="Base output directory (mirrors input directory structure).")
    parser.add_argument("--batch-dir", default="batches/judge_gpt",
                        help="Directory for intermediate batch JSONL files.")
    parser.add_argument("--debug", action="store_true",
                        help="Limit each file to the first 10 prompts.")

    args = parser.parse_args()
    logging.info(f"CLI arguments: {vars(args)}")

    secrets = load_secrets()
    client = OpenAI(api_key=secrets["openai"]["api_key"])

    # --cancel
    if args.cancel:
        if not args.batch_job_id:
            parser.error("--cancel requires --batch-job-id")
        client.batches.cancel(args.batch_job_id)
        print(f"Cancelled job: {args.batch_job_id}")

    # --poll
    elif args.poll:
        if not args.batch_job_id:
            parser.error("--poll requires --batch-job-id")
        poll_until_done(client, args.batch_job_id)

    # --retrieve
    elif args.retrieve:
        if not args.batch_job_id or not args.raw_results:
            parser.error("--retrieve requires --batch-job-id and --raw-results")
        job = client.batches.retrieve(args.batch_job_id)
        if job.status != "completed":
            print(f"Job is not yet completed (status={job.status})")
            return
        download_result(client, job.output_file_id, args.raw_results)

    # --merge
    elif args.merge:
        if not args.input_file or not args.raw_results or not args.output_file:
            parser.error("--merge requires --input-file, --raw-results, and --output-file")
        with open(args.input_file, "r", encoding="utf-8") as f:
            debates = [json.loads(line) for line in f]
        decisions = parse_decisions(args.raw_results)
        save_results(debates, decisions, args.output_file)

    # --run  (submit one file, print job ID for later polling)
    elif args.run:
        if not args.input_file or not args.batch_file:
            parser.error("--run requires --input-file and --batch-file")
        with open(args.input_file, "r", encoding="utf-8") as f:
            debates = [json.loads(line) for line in f]
        requests = build_batch_requests(debates, debug=args.debug)
        write_batch_jsonl(requests, args.batch_file)
        file_id = upload_file(client, args.batch_file)
        job = submit_batch(client, file_id)
        print(f"Job submitted. Use --poll --batch-job-id {job.id} to check status.")

    # default: full pipeline over all files in input_dirs
    else:
        file_pairs = collect_input_files(args.input_dirs, args.output_dir)
        if not file_pairs:
            print("No new files to process. Exiting.")
            return

        print(f"\nFound {len(file_pairs)} file(s) to process.")
        print(f"Model: {MODEL}\n{'=' * 60}")

        for i, (input_path, output_path) in enumerate(file_pairs, 1):
            print(f"\n[{i}/{len(file_pairs)}]")
            process_file(client, input_path, output_path, args.batch_dir, args.debug)

        print(f"\nAll done. Processed {len(file_pairs)} file(s).")


if __name__ == "__main__":
    main()
