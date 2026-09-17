import json

import numpy as np
import pytest
import torch

from forgeline.core.errors import DatasetError, MalformedRecordError, TokenizerError
from forgeline.data.collators import IGNORE_INDEX, collate_supervised, pad_sequences
from forgeline.data.preference import PreferenceDataset
from forgeline.data.pretraining import MemmapCorpus, detect_token_dtype
from forgeline.data.records import PreferenceRecord, ReadReport, SFTRecord, VerifiableTask, deterministic_split, read_jsonl, write_jsonl
from forgeline.data.streaming import ShardedTokenDataset, pack_documents, shard_token_file, write_shards_from_documents
from forgeline.data.supervised import SupervisedDataset
from forgeline.data.tokenizers import CharTokenizer, load_tokenizer, resolve_tokenizer, save_tokenizer_metadata
from forgeline.data.verifiable import extract_gsm8k_answer, load_prompts


def test_char_tokenizer_roundtrip(tmp_path):
    tok = CharTokenizer().fit("hello world")
    assert tok.decode(tok.encode("hello")) == "hello"
    save_tokenizer_metadata(tok, tmp_path)
    assert load_tokenizer(tmp_path).vocab_size == tok.vocab_size
    assert resolve_tokenizer("auto", tmp_path).kind == "char"
    with pytest.raises(TokenizerError):
        load_tokenizer(tmp_path / "missing")


def test_memmap_corpus(corpus_dir):
    c = MemmapCorpus(corpus_dir, "train", 16)
    x, y = c.sample_batch(4)
    assert x.shape == (4, 16) and torch.equal(x[:, 1:], y[:, :-1])
    assert detect_token_dtype(corpus_dir) == np.uint16
    with pytest.raises(DatasetError):
        MemmapCorpus(corpus_dir, "nope", 16)


def test_deterministic_split():
    items = list(range(100))
    a1, v1 = deterministic_split(items, 0.2, seed=3)
    a2, v2 = deterministic_split(items, 0.2, seed=3)
    assert a1 == a2 and v1 == v2 and len(v1) == 20 and not set(a1) & set(v1)


def test_jsonl_schema_validation(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"prompt": "a", "response": "b"}) + "\n" + json.dumps({"prompt": "a"}) + "\n")
    with pytest.raises(MalformedRecordError):
        read_jsonl(p, SFTRecord)
    rep = ReadReport()
    rows = read_jsonl(p, SFTRecord, on_error="skip", report=rep)
    assert len(rows) == 1 and rep.skipped == 1
    p.write_text("{not json\n")
    with pytest.raises(MalformedRecordError):
        read_jsonl(p, SFTRecord)
    with pytest.raises(MalformedRecordError):
        PreferenceRecord.from_dict({"prompt": "p", "chosen": "x", "rejected": "x"})
    with pytest.raises(MalformedRecordError):
        VerifiableTask.from_dict({"prompt": "p"})
    assert VerifiableTask.from_dict({"problem": "p", "answer": 5}).answer == "5"


def test_write_and_read_jsonl(tmp_path):
    p = tmp_path / "x.jsonl"
    n = write_jsonl(p, [SFTRecord("p", "r"), {"prompt": "a", "response": "b"}])
    assert n == 2 and len(read_jsonl(p, SFTRecord)) == 2


def test_collate_supervised_masks_prompt():
    batch = collate_supervised([([1, 2, 3], [4, 5]), ([1], [2, 3, 4, 5])], pad_value=0)
    assert batch.input_ids.shape == (2, 4)
    assert batch.targets[0, :2].tolist() == [IGNORE_INDEX, IGNORE_INDEX] and batch.targets[0, 2:].tolist() == [4, 5]
    assert batch.targets[1].tolist() == [2, 3, 4, 5]
    ids, mask = pad_sequences([[1, 2], [3]], 0, side="left")
    assert ids.tolist() == [[1, 2], [0, 3]] and mask.tolist() == [[1, 1], [0, 1]]


def test_supervised_and_preference_datasets(tokenizer, sft_jsonl, preference_jsonl):
    ds = SupervisedDataset.from_jsonl(sft_jsonl, tokenizer, max_length=32)
    assert len(ds) == 40 and all(len(p) + len(r) <= 32 for p, r in ds)
    pd = PreferenceDataset.from_jsonl(preference_jsonl, tokenizer, 16, 8)
    assert len(pd) == 40 and set(pd[0]) == {"prompt", "chosen", "rejected"}


def test_streaming_pack_and_shards(tmp_path, tokenizer):
    docs = ["hello world this is a document" for _ in range(20)]
    windows = list(pack_documents(docs, tokenizer, 8))
    assert windows and windows[0][0].shape == (8,)
    out = write_shards_from_documents(docs, tokenizer, tmp_path / "shards", max_tokens=200, shard_size=64, dtype=np.uint16,
                                      min_chars=0, drop_token_zero=False)
    assert out["num_shards"] >= 2 and (tmp_path / "shards" / "train.bin").exists()
    ids = np.arange(100, dtype=np.uint16)
    ids.tofile(tmp_path / "big.bin")
    paths = shard_token_file(tmp_path / "big.bin", tmp_path / "s", 4)
    assert len(paths) == 4
    ds = ShardedTokenDataset(tmp_path / "s", block_size=4, shuffle=False)
    x, y = next(iter(ds))
    assert x.tolist() == [0, 1, 2, 3] and y.tolist() == [1, 2, 3, 4]


def test_prompts_and_gsm8k_answer(tmp_path, tokenizer):
    p = tmp_path / "p.jsonl"
    p.write_text(json.dumps({"prompt": "hello"}) + "\nplain text line\n")
    assert len(load_prompts(p, tokenizer)) == 2
    assert len(load_prompts(None, tokenizer)) == 4
    assert extract_gsm8k_answer("some steps\n#### 1,234") == "1234"
