"""Inference engine: scheduler + paged allocator + per-sequence KV caches.

``step()`` admits pending requests (prefill), then decodes one token for every
running sequence. Sequences whose caches have the same length are decoded in
one batched forward pass; others are decoded individually. Finished sequences
release their blocks immediately so new requests can join mid-flight
(continuous batching).
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from forgeline.core.errors import ServingError
from forgeline.inference.paged_memory import OutOfBlocksError, PagedBlockAllocator
from forgeline.inference.requests import GenerationRequest, SamplingParams
from forgeline.inference.scheduler import Scheduler
from forgeline.models.generation.sampling import apply_sampling_filters, sample_from_logits
from forgeline.models.transformer.model import TransformerLM
from forgeline.observability.logging import get_logger
from forgeline.observability.tracing import Tracer

log = get_logger("forgeline.inference")


class _SequenceState:
    __slots__ = ("req", "cache", "position", "last_logits", "tokens")

    def __init__(self, req: GenerationRequest):
        self.req = req
        self.cache: Optional[List[Any]] = None
        self.position = 0
        self.last_logits: Optional[torch.Tensor] = None
        self.tokens: List[int] = list(req.prompt_tokens)


def _cat_cache(entries: List[Any]) -> Any:
    if isinstance(entries[0], tuple):
        return tuple(torch.cat([e[i] for e in entries], dim=0) for i in range(len(entries[0])))
    return torch.cat(entries, dim=0)


def _split_cache(cache: Any, n: int) -> List[Any]:
    if isinstance(cache, tuple):
        parts = [c.chunk(n, dim=0) for c in cache]
        return [tuple(p[i] for p in parts) for i in range(n)]
    return list(cache.chunk(n, dim=0))


class InferenceEngine:
    def __init__(self, model: TransformerLM, *, max_batch: int = 8, max_seq_len: Optional[int] = None, block_size: int = 16,
                 max_blocks: Optional[int] = None, device: Optional[torch.device] = None, eos_token_id: Optional[int] = None):
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.max_seq_len = min(max_seq_len or model.spec.block_size, model.spec.block_size)
        max_blocks = max_blocks or max_batch * ((self.max_seq_len + block_size - 1) // block_size + 1)
        self.allocator = PagedBlockAllocator(block_size=block_size, max_blocks=max_blocks)
        self.scheduler = Scheduler(self.allocator, max_batch=max_batch, max_seq_len=self.max_seq_len)
        self.eos_token_id = eos_token_id
        self.states: Dict[str, _SequenceState] = {}
        self.tracer = Tracer()
        self.total_tokens = 0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # ── submission ───────────────────────────────────────────────────────
    def submit(self, prompt_tokens: List[int], params: Optional[SamplingParams] = None, **kwargs: Any) -> GenerationRequest:
        params = params or SamplingParams()
        params.validate()
        if not prompt_tokens:
            raise ServingError("prompt must contain at least one token")
        req = GenerationRequest(prompt_tokens=list(prompt_tokens), params=params, **kwargs)
        self.scheduler.submit(req)
        return req

    # ── one engine step ──────────────────────────────────────────────────
    @torch.no_grad()
    def step(self) -> int:
        """Admit + prefill new requests, decode one token for running ones. Returns tokens produced."""
        with self._lock:
            produced = 0
            for req in self.scheduler.admit():
                produced += self._prefill(req)
            groups: Dict[int, List[_SequenceState]] = defaultdict(list)
            for st in list(self.states.values()):
                if not st.req.finished:
                    groups[st.position].append(st)
            for _, seqs in groups.items():
                produced += self._decode_group(seqs)
            self.total_tokens += produced
            return produced

    def _prefill(self, req: GenerationRequest) -> int:
        st = _SequenceState(req)
        self.states[req.id] = st
        req.started_at = time.time()
        try:
            with self.tracer.span("prefill", tokens=len(req.prompt_tokens)):
                ids = torch.tensor(req.prompt_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
                logits, cache = self.model.prefill(ids)
            st.cache, st.position, st.last_logits = cache, ids.shape[1], logits
            return self._emit(st)
        except Exception as exc:  # noqa: BLE001
            self._fail(st, exc)
            return 0

    def _emit(self, st: _SequenceState) -> int:
        """Sample from ``st.last_logits`` and append; retire when finished."""
        req = st.req
        p = req.params
        past = st.tokens if p.repetition_penalty != 1.0 else None
        logits = apply_sampling_filters(st.last_logits, p.temperature, p.top_k, p.top_p, p.min_p, p.repetition_penalty, past)
        token = int(sample_from_logits(logits, p.temperature).item())
        req.push_token(token)
        st.tokens.append(token)
        stop = set(p.stop_token_ids)
        if self.eos_token_id is not None:
            stop.add(self.eos_token_id)
        if token in stop:
            self._retire(st, "stop")
        elif len(req.generated_tokens) >= p.max_tokens:
            self._retire(st, "length")
        elif req.meta.get("cancel"):
            self._retire(st, "cancelled")
        elif st.position + 1 >= self.max_seq_len:
            self._retire(st, "length")
        else:
            try:
                self.allocator.append_token(req.id)
            except OutOfBlocksError as exc:
                self._fail(st, exc)
        return 1

    def _decode_group(self, seqs: List[_SequenceState]) -> int:
        try:
            with self.tracer.span("decode", batch=len(seqs)):
                next_ids = torch.tensor([[s.tokens[-1]] for s in seqs], dtype=torch.long, device=self.device)
                if len(seqs) == 1:
                    logits, cache = self.model.step(next_ids, seqs[0].cache, seqs[0].position)
                    caches = [cache]
                else:
                    merged = [_cat_cache([s.cache[i] for s in seqs]) for i in range(len(seqs[0].cache))]
                    logits, new_cache = self.model.step(next_ids, merged, seqs[0].position)
                    per_layer = [_split_cache(layer, len(seqs)) for layer in new_cache]
                    caches = [[per_layer[l][i] for l in range(len(per_layer))] for i in range(len(seqs))]
        except Exception as exc:  # noqa: BLE001
            for s in seqs:
                self._fail(s, exc)
            return 0
        produced = 0
        for i, st in enumerate(seqs):
            st.cache, st.position, st.last_logits = caches[i], st.position + 1, logits[i : i + 1]
            produced += self._emit(st)
        return produced

    def _retire(self, st: _SequenceState, reason: str) -> None:
        self.states.pop(st.req.id, None)
        st.cache = None
        self.scheduler.retire(st.req, reason)

    def _fail(self, st: _SequenceState, exc: BaseException) -> None:
        log.error("sequence_failed", request=st.req.id, error=repr(exc))
        self.states.pop(st.req.id, None)
        self.scheduler.retire(st.req, "error", error=f"{type(exc).__name__}: {exc}")

    # ── blocking helpers ─────────────────────────────────────────────────
    def run_until_idle(self, max_steps: int = 10_000) -> None:
        for _ in range(max_steps):
            if not self.scheduler.has_work():
                return
            self.step()
        raise ServingError("engine did not become idle within max_steps")

    def generate(self, prompt_tokens: List[int], params: Optional[SamplingParams] = None) -> GenerationRequest:
        req = self.submit(prompt_tokens, params)
        while not req.finished:
            self.step()
        return req

    # ── background loop (serving) ────────────────────────────────────────
    def start(self, poll_interval: float = 0.005) -> None:
        if self._thread is not None:
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                if self.scheduler.has_work():
                    self.step()
                else:
                    time.sleep(poll_interval)

        self._thread = threading.Thread(target=loop, daemon=True, name="forgeline-engine")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def stats(self) -> Dict[str, Any]:
        return {**self.scheduler.stats(), "total_tokens": self.total_tokens, "spans": self.tracer.summary()}
