"""Paged KV-cache block accounting.

Sequences reserve fixed-size blocks (pages) as they grow and release them on
completion, so memory is bounded by ``max_blocks × block_size`` tokens rather
than ``max_batch × max_seq_len``. This allocator tracks block ownership and
capacity; the engine consults it before admitting or extending a sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from forgeline.core.errors import ServingError


class OutOfBlocksError(ServingError):
    hint = "Increase max_blocks / reduce max_batch, or wait for running sequences to finish."


@dataclass
class PagedBlockAllocator:
    block_size: int = 16
    max_blocks: int = 1024
    free_blocks: List[int] = field(default_factory=list)
    tables: Dict[str, List[int]] = field(default_factory=dict)
    lengths: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.block_size < 1 or self.max_blocks < 1:
            raise ServingError("block_size and max_blocks must be >= 1")
        self.free_blocks = list(range(self.max_blocks - 1, -1, -1))

    def blocks_needed(self, n_tokens: int) -> int:
        return (n_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, seq_id: str, n_tokens: int) -> bool:
        have = len(self.tables.get(seq_id, []))
        return self.blocks_needed(n_tokens) - have <= len(self.free_blocks)

    def allocate(self, seq_id: str, n_tokens: int) -> List[int]:
        """Ensure ``seq_id`` owns enough blocks for ``n_tokens``; returns its block table."""
        table = self.tables.setdefault(seq_id, [])
        need = self.blocks_needed(n_tokens) - len(table)
        if need > len(self.free_blocks):
            raise OutOfBlocksError(f"sequence {seq_id} needs {need} more blocks, {len(self.free_blocks)} free")
        for _ in range(max(need, 0)):
            table.append(self.free_blocks.pop())
        self.lengths[seq_id] = n_tokens
        return table

    def append_token(self, seq_id: str) -> None:
        self.allocate(seq_id, self.lengths.get(seq_id, 0) + 1)

    def free(self, seq_id: str) -> None:
        for b in self.tables.pop(seq_id, []):
            self.free_blocks.append(b)
        self.lengths.pop(seq_id, None)

    @property
    def used_blocks(self) -> int:
        return self.max_blocks - len(self.free_blocks)

    @property
    def utilization(self) -> float:
        return self.used_blocks / self.max_blocks

    def stats(self) -> Dict[str, float]:
        return {"used_blocks": self.used_blocks, "free_blocks": len(self.free_blocks), "utilization": self.utilization,
                "sequences": len(self.tables)}
