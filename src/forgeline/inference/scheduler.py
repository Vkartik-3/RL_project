"""Continuous-batching scheduler: admission, running set, completion."""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Dict, List, Optional

from forgeline.inference.paged_memory import PagedBlockAllocator
from forgeline.inference.requests import GenerationRequest, RequestStatus


class Scheduler:
    """FIFO admission bounded by ``max_batch`` and the paged allocator's capacity."""

    def __init__(self, allocator: PagedBlockAllocator, max_batch: int = 8, max_seq_len: int = 512):
        self.allocator = allocator
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.pending: Deque[GenerationRequest] = deque()
        self.running: Dict[str, GenerationRequest] = {}
        self.lock = threading.Lock()
        self.completed = 0
        self.rejected = 0

    def submit(self, req: GenerationRequest) -> None:
        with self.lock:
            self.pending.append(req)

    def admit(self) -> List[GenerationRequest]:
        """Move pending requests into the running set while capacity allows."""
        admitted: List[GenerationRequest] = []
        with self.lock:
            while self.pending and len(self.running) < self.max_batch:
                req = self.pending[0]
                total = len(req.prompt_tokens) + req.params.max_tokens
                if total > self.max_seq_len:
                    self.pending.popleft()
                    req.finish("length", error=f"prompt+max_tokens {total} exceeds max_seq_len {self.max_seq_len}")
                    self.rejected += 1
                    continue
                if not self.allocator.can_allocate(req.id, len(req.prompt_tokens) + 1):
                    break  # wait for blocks to free up
                self.pending.popleft()
                self.allocator.allocate(req.id, len(req.prompt_tokens))
                req.status = RequestStatus.RUNNING
                self.running[req.id] = req
                admitted.append(req)
        return admitted

    def retire(self, req: GenerationRequest, reason: str, error: Optional[str] = None) -> None:
        with self.lock:
            self.running.pop(req.id, None)
            self.allocator.free(req.id)
            req.finish(reason, error)
            self.completed += 1

    def cancel(self, request_id: str) -> bool:
        with self.lock:
            for i, r in enumerate(self.pending):
                if r.id == request_id:
                    del self.pending[i]
                    r.status = RequestStatus.CANCELLED
                    r.finish("cancelled")
                    return True
            r = self.running.get(request_id)
            if r is not None:
                r.meta["cancel"] = True
                return True
        return False

    def has_work(self) -> bool:
        with self.lock:
            return bool(self.pending or self.running)

    def stats(self) -> Dict[str, float]:
        with self.lock:
            return {"pending": len(self.pending), "running": len(self.running), "completed": self.completed,
                    "rejected": self.rejected, **self.allocator.stats()}
