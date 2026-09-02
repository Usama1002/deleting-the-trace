# Control-token injection attacks on tool-using agents

Code and logged measurements for a study of two input-level attacks on tool-using language model agents. The first attack appends a short string of a model's own channel-control tokens to untrusted input; the tokenizer reads it as an already-closed reasoning channel, so the model emits no chain-of-thought and proceeds directly to the tool call. This deletes the reasoning trace a monitor depends on and, on requests the model would otherwise refuse, converts refusals into completed actions. The second result is that whether an identical tool-call generation actually fires is decided by the harness parser rather than the model, so agent robustness is a joint property of the model and its decoding and parsing harness.

Every experiment runs at full precision (bfloat16) with greedy decoding through the released tool sandbox. No competition leaderboard or grader scores are used as evidence; the logged JSON under `results/` is produced entirely by the scripts here.

Paper: "Control-Token Injection Suppresses Chain-of-Thought and Defeats Reasoning-Based Oversight in Tool-Using Agents" (under review at TMLR).

## Findings

1. Control-token injection suppresses the reasoning channel while the unsafe tool call still fires, defeats content-reading chain-of-thought monitors, and bypasses the model's own refusals. An empty-reasoning tripwire catches the basic attack but is defeated by a one-line benign decoy.
2. Harness parser leniency gates tool-call firing model-independently. A truncation-tolerant parser fires a call whose closing token is missing while a strict parser drops it; holding one model and its greedy decode fixed, two shipped parsers produce opposite security outcomes.

## Requirements

- One CUDA GPU with enough memory for the target models at bfloat16 (the runs were produced on a single NVIDIA H200, 141 GB).
- Python 3.12, PyTorch with CUDA, and the pinned packages in `requirements.txt` (aicomp-sdk 3.1.2, transformers 5.16.1, gymnasium 0.29, openai, openai-harmony).
- Access to the target models on Hugging Face: `openai/gpt-oss-20b`, `google/gemma-4-26B-A4B-it`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, `Qwen/Qwen3-4B-Thinking-2507`.

Setup:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

# download the target models (needs an authenticated Hugging Face CLI)
hf download openai/gpt-oss-20b
hf download google/gemma-4-26B-A4B-it
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
hf download Qwen/Qwen3-4B-Thinking-2507
```

`gpt-oss-20b` runs under transformers 4.57 or 5.16; `gemma-4-26B-A4B-it` is a multimodal model that requires transformers 5.16. The gpt-oss reasoning-suppression measurement is identical under both versions.

## Running the experiments

All experiment scripts import the shared harness, so run them with `experiments` on the path. Each script writes its output to `results/`.

```bash
export PYTHONPATH=experiments

# reproduction gate: the forge empties the gpt-oss analysis channel while the call still fires
python scripts/h200_gate.py

# E1: chain-of-thought suppression at scale
python experiments/e1_cot_suppression.py

# E2: monitor evasion and refusal bypass (generate traces, then score three monitors)
python experiments/e2b_generate.py
JUDGE=gemma python experiments/e2b_judge.py

# E3: parser leniency
python experiments/e3a_parser_leniency.py                 # deterministic, no model
MODEL=gpt_oss python experiments/e3b_end2end.py
MODEL=gemma   python experiments/e3b_end2end.py
python experiments/e3c_gemma_parser_ab.py                 # same model, two parsers

# E4: generality across reasoning models
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B TAG=deepseek python experiments/e4_generality.py
MODEL=Qwen/Qwen3-4B-Thinking-2507 TAG=qwen3 python experiments/e4_generality.py

# E5: defenses
python experiments/e5_defenses.py                         # input sanitization
python experiments/e5_d2_parser_hardening.py              # parser hardening, deterministic

# W1 to W3: ablation, token-level grounding, indirect delivery, cross-stack parser
python experiments/w1_forge_variants.py
python experiments/w1_tokenlevel.py
python experiments/w2_indirect.py
python experiments/w3_truncation_sweep.py

# adaptive empty-reasoning evasion (raw traces, effort sweep, decoy attack, monitor matrix)
python experiments/ad_traces_effort.py
python experiments/ad_decoy.py
python experiments/ad_decoy_judge.py

# regenerate the results figure from the logged JSON
python scripts/make_results_fig.py
```

## Repository structure

```
experiments/harness.py            shared harness: build an agent, run a message through the
                                  sandbox gym env, and read back the reasoning channel, the
                                  fired tool events, and the predicates
experiments/e1_*.py               chain-of-thought suppression at scale
experiments/e2*_*.py              monitor evasion and refusal bypass (generate, then judge)
experiments/e3*_*.py              parser leniency: deterministic, end-to-end, and A/B
experiments/e4_*.py               generality across reasoning models
experiments/e5_*.py               defenses: sanitization and parser hardening
experiments/w1_*.py               ablation of the injected string and token-level grounding
experiments/w2_*.py               indirect delivery and broader action types
experiments/w3_*.py               truncation robustness of the attack
experiments/ad_*.py               adaptive empty-reasoning evasion and raw traces
experiments/*_probe.py, *_debug.py supporting probes used while building the study
scripts/h200_gate.py              the reproduction gate
scripts/make_results_fig.py       the results figure
results/*.json                    logged measurements produced by the scripts above
```

## Key results

| experiment | measurement |
|---|---|
| E1 (gpt-oss-20b) | reasoning channel 52.5 to 0 tokens (empty on 55 of 55 tasks), unsafe action fires 100 percent |
| E2 monitor evasion | rule and language-model monitors detect 100 percent without the injection, 0 percent with it; 39.6 percent of refusals converted to completed exfiltrations |
| E3c parser A/B (Gemma) | same model and greedy decode: regex parser fires 24 of 24, native processor parser fires 0 of 24 |
| E4 generality | gpt-oss vulnerable via input injection; Qwen3-Thinking compliance 0.08 to 1.00 under the closed-channel primitive; DeepSeek robust |
| E5 defenses | replace-with-space sanitizer neutralizes the attack; naive deletion fails on benign inputs; parser hardening drops the truncated call |
| adaptive | empty-reasoning tripwire catches the empty forge but is defeated by a one-line benign decoy that fires 100 percent |

## Notes on reproducibility

Decoding is greedy, so each rate is deterministic per input and the reported intervals reflect the finite task sample. Seed 123 and an eight-hop interaction loop match the released sandbox. The target model revisions and the transformers version are pinned in `requirements.txt` and stated in each script.

## License

MIT License. See `LICENSE`.
