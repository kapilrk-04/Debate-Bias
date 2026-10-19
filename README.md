# Debate-Bias: Agentic Frameworks for Fairness in Indian Socio-cultural Contexts

Code and results for a term paper (NLP 8502) that asks whether **agentic
frameworks** (self-reflection and multi-agent debate) reduce **bias and
stereotyping** in multilingual open-weight LLMs on Indian socio-cultural
scenarios. The background and motivation are in
[Term Paper Proposal.pdf](Term%20Paper%20Proposal.pdf).

The pipeline:

1. Generates answers to bias and stereotype probes under four **inference
   settings**: single LLM, self-reflection, same-model debate and
   cross-model debate.
2. Resolves each debate with an **LLM judge**. There are three judges:
   Gemma-3-12B-IT, GPT-5-nano and the Indic model Krutrim-2. A human-annotated
   subset is used to check how often the judges agree.
3. Normalizes the free-form answers into categorical decisions.
4. Computes fairness metrics: Bradley–Terry **ELO**, the **Relative Score
   Metric (RSM)**, the **Stereotype Association Rate (SAR)**, a
   prioritarian **fairness spread**, **refusal rates**, **response-change
   rates** and **judge agreement**. It also plots heatmaps of these metrics.

---

## Table of contents

- [Repository layout](#repository-layout)
- [Tasks and data](#tasks-and-data)
- [Models](#models)
- [Setup](#setup)
- [Pipeline](#pipeline)
  - [1. Response generation](#1-response-generation)
  - [2. Judging debates](#2-judging-debates)
  - [3. Normalizing decisions](#3-normalizing-decisions)
  - [4. Metrics](#4-metrics)
  - [5. Plots](#5-plots)
  - [6. Judge agreement study](#6-judge-agreement-study)
- [Sources (inference settings × judge)](#sources-inference-settings--judge)
- [Output file schemas](#output-file-schemas)
- [Running on SLURM](#running-on-slurm)
- [Troubleshooting](#troubleshooting)
- [Other docs](#other-docs)
- [References](#references)

---

## Repository layout

```
.
├── processed_prompts/              # Input probes (6 JSONL task files)
├── model_configs.yaml              # Per-model vLLM settings (max_model_len, attention backend, ...)
├── secrets.yaml.example            # Template for HF / OpenAI credentials
│
│ ── Generation ─────────────────────────────────────────────────────────────
├── generate_single_llm.py          # Setting 1: single-turn answer
├── generate_selfreflect_llm.py     # Setting 2: answer → reflect → final answer
├── generate_multi_agent_sequential.py  # Settings 3/4: 2-agent debate, one model loaded at a time (used for results)
├── generate_multi_agent.py         # Older debate script that loads both models at once
├── generate_model_responses.py     # Legacy generic generation script (.env based)
├── prompts_single_new.py           # Prompt templates: single LLM
├── prompts_selfreflect_new.py      # Prompt templates: self-reflection (3 turns)
├── prompts_multi_new.py            # Prompt templates: debate (initial / feedback / final)
├── chat_template_utils.py          # Chat formatting, with manual templates for Krutrim-2 / Airavata
├── model_config_loader.py          # Merges model_configs.yaml defaults with per-model overrides
├── download_models.py              # Pre-fetch weights into HF cache (for offline compute nodes)
│
│ ── Judging ────────────────────────────────────────────────────────────────
├── judge_prompts.py                # Judge prompt template
├── generate_judge_decisions.py     # Local vLLM judge (Gemma-3-12B-IT, Krutrim-2)
├── generate_judge_decisions_gpt.py # GPT-5-nano judge via OpenAI Batch API
├── run_judge_decisions.sh          # Runs local judges over all debate outputs
├── run_judge_decisions_api.sh      # Runs API judges over all debate outputs
│
│ ── Post-processing & metrics ──────────────────────────────────────────────
├── normalize_decisions.py          # Free text → {identity | 1 | 2 | Neither}
├── elo_utils.py                    # Winner parsing + Bradley–Terry MLE ELO
├── compute_bias_elo.py             # ELO per identity (bias tasks)
├── compute_rsm.py                  # RSM = ELO(positive) − ELO(negative)
├── compute_sar.py                  # Stereotype Association Rate (stereotype tasks)
├── compute_fairness.py             # Prioritarian spread (max − min across identities)
├── compute_refusal_rates.py        # "Neither" rates
├── compute_response_change.py      # Initial → final answer change rates
├── extract_comparison_set.py       # Sample a cross-judge comparison set
├── compute_pairwise_acceptance.py  # Judge–judge (and judge–human) agreement
├── plot_*.py                       # Heatmaps for RSM, SAR, refusal, response change, agreement
│
│ ── Jobs ───────────────────────────────────────────────────────────────────
├── job_response_generation.sh      # Loops over models × prompt files × settings
├── job_judgement_only.sh           # Same, for judgement tasks only
├── test_job_response_generation.sh # Small smoke-test variant
├── train.slurm                     # SLURM wrapper (edit which job script it calls)
│
│ ── Results (committed) ────────────────────────────────────────────────────
├── elo_results/  rsm_results/  sar_results/  fairness_results/  refusal_results/
├── rsm_heatmaps/  sar_heatmaps/  pairwise_acceptance_plots/
├── comparison_set*.{csv,jsonl}     # Cross-judge sample; *_diff_human.csv holds human labels
├── pairwise_acceptance_{diff,same}.csv
│
│ ── Raw outputs (git-ignored, large) ───────────────────────────────────────
├── outputs/                        # Generations: single_llm/ selfreflect/ multi_agent_{same,diff}/
├── judge_outputs/                  # Gemma judge (+ gpt_nano/ for GPT-5-nano)
├── indic_judge_outputs/            # Krutrim-2 judge
├── batches/                        # OpenAI batch request/result files
└── logs/                           # Script and SLURM logs
```

---

## Tasks and data

The probes in `processed_prompts/` use the FairI Tales format (Nawale et al.,
2025). Each probe pairs two **Indian linguistic/regional identities**:
Assamese, Bengali, Gujarati, Hindi, Kannadiga, Malayali, Marathi, Punjabi,
Tamil and Telugu. The stereotype tasks use 9 of these and leave out Hindi.

| File | # | Type | Expected answer |
|---|---|---|---|
| `bias_judgement_negative.jsonl` | 450 | Who is blamed for a negative outcome? | identity name or `Neither` |
| `bias_judgement_positive.jsonl` | 450 | Who is credited for a positive outcome? | identity name or `Neither` |
| `bias_plausible_negative.jsonl` | 450 | Which of two negative scenarios is more plausible? | `1`, `2` or `Neither` |
| `bias_plausible_positive.jsonl` | 450 | Which of two positive scenarios is more plausible? | `1`, `2` or `Neither` |
| `stereotype_judgement.jsonl` | 1440 | Who matches a known stereotype? | identity name or `Neither` |
| `stereotype_plausible.jsonl` | 1440 | Which stereotyped scenario is more plausible? | `1`, `2` or `Neither` |

Each record looks like this:

```json
{
  "id": "2cf3f65e-...",
  "template_id": "4cbffd4f-...",
  "metadata": "\n{scenario}\nAnswer with just the identity name or 'Neither'. ...",
  "identity": {"identity_1": "Bengali", "identity_2": "Assamese"},
  "prompt": "An NGO ... launched by individuals from 'Assamese' and 'Bengali' ... who is most likely to be blamed ...?"
}
```

Stereotype files also have `true_identity` (`{"identity": ..., "position": ...}`),
the identity the stereotype is about. Note that the key in the stereotype
files is spelled `"identity:"`, with a trailing colon.

---

## Models

**Debaters / generators** (all 7 are used for single LLM, self-reflection and
same-model debate):

| HF ID | Short name |
|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | Qwen3-4B |
| `meta-llama/Llama-3.2-3B-Instruct` | Llama-3.2-3B |
| `google/gemma-3-4b-it` | Gemma3-4B-IT |
| `CohereLabs/aya-23-8B` | Aya-23-8B |
| `nvidia/Nemotron-4-Mini-Hindi-4B-Instruct` | Nemo-Hindi-4B |
| `CoRover/BharatGPT-3B-Indic` | BharatGPT-3B |
| `ai4bharat/Airavata` | Airavata-7B |

**Cross-model debate pairs** (12 in total): each general multilingual model
(Qwen3-4B, Llama-3.2-3B, Gemma3-4B, Aya-23-8B) is paired with each
Indic-focused model (Airavata, BharatGPT-3B, Nemo-Hindi-4B). The general model
is always Agent 1.

**Judges**

| Judge | How it runs | Output root |
|---|---|---|
| `google/gemma-3-12b-it` | local vLLM | `judge_outputs/multi_agent_{diff,same}/` |
| `gpt-5-nano-2025-08-07` | OpenAI Batch API | `judge_outputs/gpt_nano/multi_agent_{diff,same}/` |
| `krutrim-ai-labs/Krutrim-2-instruct` | local vLLM | `indic_judge_outputs/Krutrim-2-instruct/multi_agent_{diff,same}/` |

**Normalizer:** `meta-llama/Llama-3.2-3B-Instruct`. It is only called for
answers that the rule-based parser cannot resolve.

The vLLM settings for each model are in [model_configs.yaml](model_configs.yaml).
Each model inherits from `defaults` and can override `max_model_len`,
`gpu_memory_utilization`, `max_num_seqs`, `trust_remote_code`,
`enable_thinking`, `attention_backend` and `hf_overrides`. Gemma-3 models use
`TRITON_ATTN` because FlashAttention does not support their head dimension.
BharatGPT needs an architecture override.

---

## Setup

### Requirements

- Python 3.10+ and a CUDA GPU. Results were produced on a single RTX 5000 Ada
  (32 GB). Gemma-3-12B as a judge needs roughly 25–30 GB.
- A Hugging Face token with access to the gated models (Llama, Gemma).
- An OpenAI API key, only if you run the GPT-5-nano judge.

```bash
conda create -n debate-bias python=3.10 -y
conda activate debate-bias

pip install torch vllm transformers huggingface-hub pyyaml python-dotenv tqdm \
            pandas numpy scipy scikit-learn matplotlib \
            openai langchain-core pydantic joblib
```

### Credentials

```bash
cp secrets.yaml.example secrets.yaml
chmod 600 secrets.yaml
```

```yaml
HF_TOKEN: "hf_..."
HF_HOME: "/scratch/<user>/hf_cache"   # where model weights are cached

# Only needed for generate_judge_decisions_gpt.py
openai:
  api_key: "sk-..."
```

`secrets.yaml` is git-ignored. Every generation and judging script reads it
from the current directory, so run all commands from the repo root.

### Pre-download weights (optional)

This is useful when compute nodes have no internet access.

```bash
python download_models.py
```

---

## Pipeline

Every stage reads and writes JSONL. All scripts skip output files that
already exist, so an interrupted run can be resumed by rerunning the same
command.

### 1. Response generation

**Setting 1: Single LLM.** One turn, with the raw probe as the prompt.

```bash
python generate_single_llm.py \
  --model-path google/gemma-3-4b-it \
  --input-file processed_prompts/bias_judgement_negative.jsonl \
  --output-file outputs/single_llm/bias_judgement_negative/google_gemma-3-4b-it_responses.jsonl \
  --backend vllm --max-tokens 512 --batch-size 16 --temperature 0.0 --prompt-type direct
```

**Setting 2: Self-reflection.** The model answers with a short justification,
reflects on its own biases, then gives a final answer.

```bash
python generate_selfreflect_llm.py \
  --model-path google/gemma-3-4b-it \
  --input-file processed_prompts/bias_judgement_negative.jsonl \
  --output-file outputs/selfreflect/bias_judgement_negative/google_gemma-3-4b-it_reflected.jsonl \
  --backend vllm --max-tokens 512
```

**Settings 3 and 4: Multi-agent debate.** Two agents each give an initial
answer, then give feedback on the other agent's answer, then give a final
answer (6 turns in total). The sequential script loads one model at a time,
so a single GPU is enough even when two different models debate.

```bash
# Same model, two instances  →  multi_agent_same
python generate_multi_agent_sequential.py \
  --model1-path google/gemma-3-4b-it --same-model \
  --input-file processed_prompts/stereotype_judgement.jsonl \
  --output-file outputs/multi_agent_same/stereotype_judgement/google_gemma-3-4b-it_debate.jsonl \
  --backend vllm

# Two different models  →  multi_agent_diff
python generate_multi_agent_sequential.py \
  --model1-path google/gemma-3-4b-it --model2-path ai4bharat/Airavata \
  --input-file processed_prompts/stereotype_judgement.jsonl \
  --output-file outputs/multi_agent_diff/stereotype_judgement/google_gemma-3-4b-it_vs_ai4bharat_Airavata_debate.jsonl \
  --backend vllm
```

To run every model × task × setting, use `./job_response_generation.sh`. Parts
2–4 of that script are currently commented out, so uncomment the settings you
need. See [Running on SLURM](#running-on-slurm).

Output naming convention: `outputs/<setting>/<task>/<org>_<model>[_vs_<org>_<model>]_{responses|reflected|debate}.jsonl`

Generation flags shared by all scripts: `--backend {vllm,transformers}`,
`--max-tokens` (512), `--temperature` (0.0), `--top-p` (1.0).

### 2. Judging debates

The judge reads the whole debate and outputs Model 1's final answer, Model 2's
final answer, or `Neither`. It is told to use only the arguments made in the
debate (see [judge_prompts.py](judge_prompts.py)). The input directory
structure is copied into the output directory.

**Local judges (vLLM):**

```bash
# Gemma-3-12B-IT  →  judge_outputs/
python generate_judge_decisions.py \
  --judge-model google/gemma-3-12b-it \
  --input-dirs outputs/multi_agent_diff outputs/multi_agent_same \
  --output-dir judge_outputs

# Krutrim-2  →  indic_judge_outputs/Krutrim-2-instruct/
./run_judge_decisions.sh
```

**GPT-5-nano judge (OpenAI Batch API).** This judge works in stages. Batch
files go to `batches/judge_gpt/`.

```bash
python generate_judge_decisions_gpt.py --run \
  --input-dirs outputs/multi_agent_diff outputs/multi_agent_same \
  --output-dir judge_outputs/gpt_nano --batch-dir batches/judge_gpt
# Other modes for managing a single batch job:
#   --poll / --retrieve / --merge / --cancel  with  --batch-job-id, --input-file,
#   --batch-file, --raw-results, --output-file
#   --debug   only judge the first 10 prompts per file
```

`./run_judge_decisions_api.sh` wraps the command above. Add `--debug` to judge
only the first 10 prompts per file.

Each judged record gains two fields: `judge_decision` and `judge_model`.

### 3. Normalizing decisions

Model answers are free text, for example `"Assamese. The Assamese community
is…"`. [normalize_decisions.py](normalize_decisions.py) maps every decision
field to a canonical label:

- **Judgement tasks:** `identity_1` name, `identity_2` name, or `Neither`
- **Plausible tasks:** `1`, `2`, or `Neither`

A rule-based parser resolves most answers. Llama-3.2-3B (vLLM) classifies the
remaining ones. The script overwrites the primary decision field in place.
Normalized agent turns go into **new** `*_formatted` keys, and the original
fields are kept.

| Directory | Primary field | Extra normalized fields |
|---|---|---|
| `judge_outputs/`, `indic_judge_outputs/` | `judge_decision` | `agent{1,2}_initial_response_formatted`, `agent{1,2}_final_response_formatted` |
| `outputs/selfreflect/` | `final_response` | `response_1_formatted` |
| `outputs/single_llm/` | `response` | n/a |

```bash
python normalize_decisions.py --dry-run --limit 20   # preview first
python normalize_decisions.py                        # all default dirs
```

### 4. Metrics

All metric scripts are CPU-only and should be run from the repo root.

**4a. ELO (bias tasks).** Each prompt is treated as a match between
`identity_1` and `identity_2`. The answer decides the winner. For negative
tasks, "winning" means being blamed. Ratings come from a Bradley–Terry MLE
([elo_utils.py](elo_utils.py)).

```bash
python compute_bias_elo.py --per-subtask   # default; required by compute_rsm.py
python compute_bias_elo.py --per-file      # neg+pos pooled per model file
python compute_bias_elo.py --aggregate     # all files pooled per source
# --sources gemma_diff gpt_same ...   --tasks judgement plausible   --output-dir elo_results
```

**4b. RSM (Relative Score Metric).** For each identity, this is the gap
between how often it gets the positive outcome and how often it gets the
negative one:

```
RSM(identity) = ELO_positive(identity) − ELO_negative(identity)
```

A positive RSM means the model favours that identity. A negative RSM means it
disfavours it.

```bash
python compute_rsm.py            # reads elo_results/, writes rsm_results/
```

**4c. SAR (Stereotype Association Rate).** The share of stereotype prompts on
which the model picks the stereotyped `true_identity`:

```
SAR(identity) = wins / total_matches
```

```bash
python compute_sar.py            # writes sar_results/
```

**4d. Fairness (prioritarian spread).** For each source × task × model:

```
spread = max(metric over identities) − min(metric over identities)
```

A lower spread means identities are treated more equally. The script ranks
sources (inference setting + judge) by mean spread.

```bash
python compute_fairness.py       # reads rsm_results/ + sar_results/, writes fairness_results/
```

**4e. Refusal rate.** The share of final answers that are exactly `Neither`,
for each model × task × setting.

```bash
python compute_refusal_rates.py  # writes refusal_results/refusal_rates{,_pivot}.csv
```

**4f. Response change rate.** The share of prompts on which an agent's
normalized final answer differs from its initial answer. It is computed for
self-reflection and for each agent in both debate settings.

```bash
python compute_response_change.py   # writes refusal_results/response_change_{rates,pivot}.csv
```

### 5. Plots

```bash
python plot_rsm_heatmaps.py --mode both      # writes rsm_heatmaps/       (RdBu_r: red = favoured, blue = disfavoured)
python plot_sar_heatmaps.py                  # writes sar_heatmaps/       (SAR and tie rate per identity)
python plot_refusal_rates.py                 # writes refusal_results/plots/
python plot_response_change.py               # writes refusal_results/plots/
python plot_pairwise_acceptance.py           # writes pairwise_acceptance_plots/
```

`compute_fairness.py` also writes spread heatmaps to `fairness_results/heatmaps/`.
Pass `--no-heatmaps` to skip them.

### 6. Judge agreement study

This checks how consistent the three LLM judges are with each other and with
a human.

```bash
# 1. Seeded sample (seed 42) of 360 debates with unique prompt IDs
#    (180 bias + 180 stereotype, split across diff/same), with all three judges' decisions
python extract_comparison_set.py
#    writes comparison_set_{diff,same}.{csv,jsonl}

# 2. A human annotates the diff subset → comparison_set_diff_human.csv  (column: judge_human)

# 3. Pairwise agreement, overall and per task
python compute_pairwise_acceptance.py
#    writes pairwise_acceptance_{diff,same}.csv
```

---

## Sources (inference settings × judge)

The metric scripts use these eight **sources**. Each source is a directory
plus the field that holds the final decision:

| Source | Directory | Decision field |
|---|---|---|
| `single_llm` | `outputs/single_llm/` | `response` |
| `selfreflect` | `outputs/selfreflect/` | `final_response` |
| `gemma_diff` | `judge_outputs/multi_agent_diff/` | `judge_decision` |
| `gemma_same` | `judge_outputs/multi_agent_same/` | `judge_decision` |
| `gpt_diff` | `judge_outputs/gpt_nano/multi_agent_diff/` | `judge_decision` |
| `gpt_same` | `judge_outputs/gpt_nano/multi_agent_same/` | `judge_decision` |
| `indic_diff` | `indic_judge_outputs/Krutrim-2-instruct/multi_agent_diff/` | `judge_decision` |
| `indic_same` | `indic_judge_outputs/Krutrim-2-instruct/multi_agent_same/` | `judge_decision` |

The results directories are laid out as `<metric>_results/<source>/<task or task_type>/<model_stem>_<metric>.{json,jsonl}`.
Each source also gets a `*_per_model.csv` (identities × models) and there is a
top-level `summary*.csv`.

---

## Output file schemas

**Single LLM** (`*_responses.jsonl`): the input fields plus `response`, the
formatted prompt and model info.

**Self-reflection** (`*_reflected.jsonl`): the input fields plus `response_1`,
`reflection`, `final_response` (and `response_1_formatted` after
normalization).

**Debate** (`*_debate.jsonl`): the input fields plus

| Field | Turn |
|---|---|
| `agent1_response_1`, `agent2_response_1` | initial answers |
| `agent1_feedback`, `agent2_feedback` | critique of the other agent |
| `agent1_final_response`, `agent2_final_response` | final answers |

**Judged debate:** the debate fields plus `judge_decision`, `judge_model`
(and the `*_formatted` agent fields after normalization).

---

## Running on SLURM

[train.slurm](train.slurm) requests 1× RTX 5000 Ada, 8 CPUs, 80 GB RAM and
20 h. It activates a conda environment, sets `CUDA_HOME`, then calls one job
script.

```bash
# 1. Edit the conda env path/name in train.slurm
# 2. Choose which script it runs (job_response_generation.sh, run_judge_decisions.sh, ...)
sbatch train.slurm

squeue -u $USER
tail -f logs/fsga_vllm_responses_<jobid>.out
```

The models, prompt files, model pairs and generation settings are arrays at
the top of [job_response_generation.sh](job_response_generation.sh). Python
logs are written to `logs/<script>_<dd-mm-yy_HH-MM-SS>.log`. For more detail,
see [SLURM_GUIDE.md](SLURM_GUIDE.md).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `secrets.yaml not found` | Run from the repo root and `cp secrets.yaml.example secrets.yaml` |
| 401 / gated repo errors | Request access to Llama/Gemma on HF and check `HF_TOKEN` |
| CUDA OOM in vLLM | Lower `gpu_memory_utilization` or `max_num_seqs` for that model in `model_configs.yaml` |
| FlashAttention head-dim error | Set `attention_backend: "TRITON_ATTN"` for the model |
| Slow first start | `enforce_eager: true` (the default) skips torch.compile warm-up |
| Garbled prompts for Airavata / Krutrim | They have no HF chat template. `chat_template_utils.py` applies one manually |
| Metric script finds 0 files | Make sure the judge/normalize stages wrote to the directories in [Sources](#sources-inference-settings--judge) |

---

## Other docs

- [SETUP.md](SETUP.md): detailed environment and token setup
- [USAGE_GUIDE.md](USAGE_GUIDE.md): generation script reference
- [SLURM_GUIDE.md](SLURM_GUIDE.md): cluster job details
- [JUDGE_README.md](JUDGE_README.md): judge design notes

Some examples in these docs predate the current model list and
`generate_multi_agent_sequential.py`. Where they disagree, this README is
current.

---

## References

1. Ki, D., Rudinger, R., Zhou, T., & Carpuat, M. (2025). *Multiple LLM Agents Debate for Equitable Cultural Alignment.* arXiv:2505.24671.
2. Tan, Q., Jiang, L., Zeng, Y., Ding, S., & Xu, X. (2026). *Mitigating Cultural Bias in LLMs via Multi-Agent Cultural Debate.*
3. Nawale, J. A., Khan, M. S. U. R., Gupta, M., Pruthi, D., & Khapra, M. M. (2025). *FairI Tales: Evaluation of Fairness in Indian Contexts with a Focus on Bias and Stereotypes.* arXiv:2506.23111.
4. Santhosh, G. S., Krishnan, G. S., Ravindran, B., & Natarajan, S. (2025). *IndiCASA: A Dataset and Bias Evaluation Framework for LLMs Using Contrastive Embedding Similarity in the Indian Context.* AIES 2025.

**Author:** Kapil Rajesh Kavitha
