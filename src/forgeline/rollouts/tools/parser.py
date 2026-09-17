"""Parse ``<think>``, ``<tool_call>``, ``<tool_result>`` and ``<final_answer>`` tags."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FINAL_ANS_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_RESULT_RE = re.compile(r"<tool_result>(.*?)</tool_result>", re.DOTALL)


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class ParsedOutput:
    tool_call: Optional[ToolCall]
    final_answer: Optional[str]
    thinking: Optional[str]
    tool_results: List[str]
    raw: str


def parse_tagged_output(text: str) -> ParsedOutput:
    tool_call = None
    m = _TOOL_CALL_RE.search(text)
    if m:
        try:
            data = json.loads(m.group(1).strip())
            if isinstance(data, dict) and "name" in data:
                tool_call = ToolCall(name=str(data["name"]), args=data.get("args", {}) or {})
        except json.JSONDecodeError:
            pass
    m = _FINAL_ANS_RE.search(text)
    final_answer = m.group(1).strip() if m else None
    m = _THINK_RE.search(text)
    thinking = m.group(1).strip() if m else None
    tool_results = [r.group(1).strip() for r in _TOOL_RESULT_RE.finditer(text)]
    return ParsedOutput(tool_call=tool_call, final_answer=final_answer, thinking=thinking,
                        tool_results=tool_results, raw=text)


def format_tool_result(text: str) -> str:
    return f"\n<tool_result>\n{text}\n</tool_result>\n"
