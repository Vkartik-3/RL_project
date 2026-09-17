# Data

## What

Tokenizers, record schemas, datasets and collators for every training stage: pretraining token corpora, SFT pairs, preference pairs, reward-model data, verifiable tasks and process-reward records.

## Why

Every downstream algorithm assumes clean, typed input. Validation at load time turns a silent training problem (an empty `chosen`, a missing answer, a label list of the wrong length) into an error that names the file and line.

## How

### Tokenizers (`data/tokenizers.py`)

| Spec | Class | Notes |
|---|---|---|
| `char` | `CharTokenizer` | fitted on a corpus, or `CharTokenizer.ascii()` (97 symbols); dependency-free |
| `tiktoken[:encoding]` | `TiktokenTokenizer` | `training` extra |
| `hf:<name>` | `HFTokenizer` | `huggingface` extra |
| `auto` | from `meta.pkl` | written next to token files and embedded in checkpoints |

Token files use `uint16` when the vocabulary fits (≤ 65,535) and `uint32` otherwise (`token_dtype_for`, `detect_token_dtype`).

### Record schemas (`data/records.py`)

| Kind | Class | Required fields |
|---|---|---|
| supervised | `SFTRecord` | `prompt`, `response` (aliases `completion`, `target`) |
| preference | `PreferenceRecord` | `prompt`, `chosen`, `rejected` (chosen ≠ rejected) |
| verifiable | `VerifiableTask` | `prompt` (alias `problem`, `question`) and `answer` or `test_code`; optional `tools`, `task_type` |
| process | `ProcessRecord` | `prompt`, `steps`, `step_labels` (same length, finite) |
| reward | `RewardRecord` | `text`, finite `score` |

`read_jsonl(path, schema, on_error="raise"|"skip", report=ReadReport())` validates every line. Extra keys are preserved in `meta`.

`deterministic_split(items, val_fraction, seed)` produces identical splits for identical inputs.

### Datasets

| Class | Kind | Behaviour |
|---|---|---|
| `MemmapCorpus` | pretraining | random `(x, y)` windows of `block_size` from `<split>.bin`; seeded via `torch.Generator` |
| `ShardedTokenDataset` | pretraining | streams `shard_*.bin`, splits shards by rank and DataLoader worker |
| `TextGlobDataset` | pretraining | tokenizes raw text files on the fly |
| `HFStreamingDataset` | pretraining | streams a HuggingFace dataset, packs tokens (`huggingface` extra) |
| `SupervisedDataset` | supervised | tokenized `(prompt_ids, response_ids)`, response truncated to fit `max_length` |
| `PreferenceDataset` | preference | tokenized prompt (left-truncated) + chosen/rejected (right-truncated) |
| `VerifiableTaskDataset` | verifiable | tasks from JSONL or the built-in arithmetic set |

### Preprocessing

* `prepare_text_corpus(input, output, tokenizer, val_fraction)` → `train.bin`, `val.bin`, `meta.pkl`.
* `write_shards_from_documents(...)` → fixed-size shards with a pre-allocated buffer (constant memory), first shard as validation, then merged `train.bin`/`val.bin`.
* `prepare_hf_corpus(...)` — the same for a streamed HuggingFace dataset.
* `shard_token_file(input, output, n)` — split an existing token file.
* `build_gsm8k_tool_dataset(output)` — GSM8K as tool-use verifiable tasks.

### Collators

`collate_supervised` produces next-token `input_ids`/`targets` with prompt tokens and padding set to `IGNORE_INDEX = -1`. `pad_sequences` supports left/right padding with an attention mask. `collate_preference` pads prompt (left) and responses (right).

## Configuration

```yaml
data:
  kind: supervised          # pretraining | supervised | preference | reward | verifiable | process
  path: data/samples/supervised/arithmetic_sft.jsonl
  tokenizer: char           # auto | char | tiktoken[:enc] | hf:<name>
  val_fraction: 0.1
  max_length: 48
  seed: 1337
  extra: {on_error: skip, max_prompt_length: 384, limit: 100}
```

CLI: `forgeline data prepare|prepare-hf|shard|gsm8k|validate`.

## Failure modes

| Condition | Error |
|---|---|
| missing file | `DatasetError` |
| invalid JSON / wrong field types / identical chosen & rejected | `MalformedRecordError` with `path:line` |
| corpus shorter than `block_size + 1` | `DatasetError` |
| missing `meta.pkl` for `tokenizer: auto` | `TokenizerError` |
| optional backend not installed | `OptionalDependencyError` naming the extra |

## Local validation

`tests/unit/test_data.py` covers tokenizers, memmap sampling, deterministic splits, schema validation, collators, packing and sharding. `forgeline data validate --input f.jsonl --kind preference` reports valid/skipped counts.

## Hardware requirements

CPU. Streaming large HuggingFace corpora needs network access and disk for shards (about 4 bytes per token with `uint32`).

## Limitations

* `CharTokenizer` drops characters outside its vocabulary on encode.
* Shard shuffling is per-epoch within a process; there is no global cross-rank shuffle buffer.
