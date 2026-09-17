import json

import pytest
import torch

from forgeline.core.config import model_spec_from_preset
from forgeline.core.errors import ServingError
from forgeline.inference.engine import InferenceEngine
from forgeline.inference.paged_memory import OutOfBlocksError, PagedBlockAllocator
from forgeline.inference.requests import GenerationRequest, RequestStatus, SamplingParams
from forgeline.inference.scheduler import Scheduler
from forgeline.models.transformer.model import TransformerLM
from forgeline.serving.api import Router
from forgeline.serving.backends import EchoBackend, EngineBackend
from forgeline.serving.schemas import ChatCompletionRequest, CompletionRequest


def test_paged_allocator():
    a = PagedBlockAllocator(block_size=4, max_blocks=3)
    a.allocate("s1", 5)
    assert a.used_blocks == 2 and a.can_allocate("s2", 4) and not a.can_allocate("s2", 5)
    with pytest.raises(OutOfBlocksError):
        a.allocate("s2", 8)
    a.append_token("s1")  # 6 tokens still fits in 2 blocks
    a.free("s1")
    assert a.used_blocks == 0 and a.utilization == 0.0


def test_scheduler_admission_and_rejection():
    s = Scheduler(PagedBlockAllocator(block_size=4, max_blocks=4), max_batch=1, max_seq_len=12)
    r1 = GenerationRequest([1, 2, 3], SamplingParams(max_tokens=2))
    r2 = GenerationRequest([1], SamplingParams(max_tokens=2))
    too_long = GenerationRequest([1] * 10, SamplingParams(max_tokens=5))
    for r in (too_long, r1, r2):
        s.submit(r)
    admitted = s.admit()
    assert too_long.status == RequestStatus.FAILED and "exceeds" in too_long.error
    assert admitted == [r1] and s.stats()["pending"] == 1
    s.retire(r1, "stop")
    assert s.admit() == [r2] and s.stats()["completed"] == 1
    with pytest.raises(ServingError):
        SamplingParams(max_tokens=0).validate()


def test_engine_continuous_batching_and_greedy_equivalence():
    m = TransformerLM(model_spec_from_preset("tiny", vocab_size=40, block_size=32)).eval()
    eng = InferenceEngine(m, max_batch=2, block_size=4)
    reqs = [eng.submit([1, 2], SamplingParams(max_tokens=4, temperature=0)), eng.submit([1, 2, 3], SamplingParams(max_tokens=6, temperature=0)),
            eng.submit([5], SamplingParams(max_tokens=3, temperature=0))]
    eng.step()
    assert eng.scheduler.stats()["running"] == 2 and eng.scheduler.stats()["pending"] == 1  # max_batch honoured
    eng.run_until_idle()
    assert all(r.status == RequestStatus.FINISHED for r in reqs) and [len(r.generated_tokens) for r in reqs] == [4, 6, 3]
    for r in reqs:
        ref = m.generate(torch.tensor([r.prompt_tokens]), r.params.max_tokens, temperature=0)[0, len(r.prompt_tokens):].tolist()
        assert r.generated_tokens == ref
    assert eng.stats()["used_blocks"] == 0 and eng.stats()["total_tokens"] == 13


def test_engine_stop_token_and_cancel():
    m = TransformerLM(model_spec_from_preset("tiny", vocab_size=40, block_size=32)).eval()
    first = m.generate(torch.tensor([[1, 2]]), 1, temperature=0)[0, -1].item()
    eng = InferenceEngine(m, eos_token_id=first)
    r = eng.generate([1, 2], SamplingParams(max_tokens=10, temperature=0))
    assert r.finish_reason == "stop" and len(r.generated_tokens) == 1
    with pytest.raises(ServingError):
        eng.submit([], SamplingParams())
    r2 = eng.submit([1, 2, 3], SamplingParams(max_tokens=10, temperature=0))
    assert eng.scheduler.cancel(r2.id) and r2.status == RequestStatus.CANCELLED


def test_schemas():
    r = CompletionRequest.from_dict({"prompt": ["hi"], "max_tokens": 3, "stop": "x"})
    assert r.prompt == "hi" and r.stop == ["x"]
    for bad in ({"prompt": ""}, {"prompt": "a", "max_tokens": 0}, {"prompt": "a", "top_p": 2}, {"prompt": "a", "stop": 3}, "nope"):
        with pytest.raises(ServingError):
            CompletionRequest.from_dict(bad)
    c = ChatCompletionRequest.from_dict({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]})
    assert c.to_prompt().endswith("<|assistant|>\n")
    with pytest.raises(ServingError):
        ChatCompletionRequest.from_dict({"messages": [{"role": "alien", "content": "x"}]})


def test_router_with_echo_backend():
    router = Router({"a": EchoBackend("a"), "b": EchoBackend("b")}, "a")
    assert router.handle("GET", "/health")[1]["status"] == "ok"
    assert {m["id"] for m in router.handle("GET", "/v1/models")[1]["data"]} == {"a", "b"}
    status, body = router.handle("POST", "/v1/completions", {"prompt": "hello", "max_tokens": 3, "model": "b"})
    assert status == 200 and body["choices"][0]["text"] == "oll" and body["model"] == "b" and body["usage"]["total_tokens"] == 8
    status, body = router.handle("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    assert status == 200 and body["choices"][0]["message"]["role"] == "assistant"
    status, chunks = router.handle("POST", "/v1/completions", {"prompt": "ab", "stream": True})
    lines = list(chunks)
    assert lines[-1] == "data: [DONE]\n\n" and json.loads(lines[0][6:])["choices"][0]["text"] == "b"
    assert router.handle("POST", "/v1/completions", {"prompt": "x", "model": "zzz"})[0] == 400
    assert router.handle("GET", "/nope")[0] == 404
    router.backends["b"].disable()
    assert router.handle("POST", "/v1/completions", {"prompt": "x", "model": "b"})[0] == 503
    assert router.handle("GET", "/metrics")[1]["requests_total"] == 3 and router.errors_total == 2


def test_engine_backend_end_to_end(policy):
    eng = InferenceEngine(policy.model, max_batch=2)
    backend = EngineBackend("tiny", eng, policy.tokenizer, background=False)
    res = backend.complete("hello", SamplingParams(max_tokens=5, temperature=0.0))
    assert res.completion_tokens == 5 and isinstance(res.text, str)
    chunks = list(backend.stream("hello", SamplingParams(max_tokens=3, temperature=0.0)))
    assert chunks and "".join(chunks) == backend.complete("hello", SamplingParams(max_tokens=3, temperature=0.0)).text
