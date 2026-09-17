"""GGUF export (FP16, Q8_0, Q4_0) for the native model.

Block-32 absmax quantization for Q8_0 / Q4_0 follows the ggml layouts:

* Q8_0: ``fp16 scale`` + 32 × int8
* Q4_0: ``fp16 scale`` + 16 bytes, low nibbles hold elements 0..15 and high
  nibbles hold elements 16..31 (ggml layout), values stored as ``q + 8``.

Tensor data offsets are written correctly (relative to the aligned start of
the data section) so the file is spec-compliant.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch

from forgeline.core.config import ModelSpec
from forgeline.core.errors import ExportError

GGUF_MAGIC = 0x46475547
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_UINT64 = 10

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8

FILE_TYPE = {GGML_TYPE_F16: 1, GGML_TYPE_Q8_0: 7, GGML_TYPE_Q4_0: 2}
QUANT_FORMATS = ("none", "f16", "q8_0", "q4_0")


def quantize_q8_0(tensor: torch.Tensor) -> Tuple[bytes, Tuple[int, ...], int]:
    arr = tensor.detach().float().cpu().numpy()
    shape = arr.shape
    flat = arr.flatten()
    n = flat.size
    pad = (32 - n % 32) % 32
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    blocks = flat.reshape(-1, 32)
    amax = np.max(np.abs(blocks), axis=1)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.round(blocks / scale[:, None]).clip(-128, 127).astype(np.int8)
    out = bytearray()
    for s, row in zip(scale, q):
        out.extend(struct.pack("<e", float(np.float16(s))))
        out.extend(row.tobytes())
    return bytes(out), shape, n


def quantize_q4_0(tensor: torch.Tensor) -> Tuple[bytes, Tuple[int, ...], int]:
    arr = tensor.detach().float().cpu().numpy()
    shape = arr.shape
    flat = arr.flatten()
    n = flat.size
    pad = (32 - n % 32) % 32
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    blocks = flat.reshape(-1, 32)
    amax = np.max(np.abs(blocks), axis=1)
    scale = np.where(amax > 0, amax / 7.0, 1.0).astype(np.float32)
    q = np.round(blocks / scale[:, None]).clip(-8, 7).astype(np.int8) + 8  # 0..15
    out = bytearray()
    for s, row in zip(scale, q):
        out.extend(struct.pack("<e", float(np.float16(s))))
        low = row[:16].astype(np.uint8)
        high = row[16:].astype(np.uint8)
        out.extend(((high << 4) | low).astype(np.uint8).tobytes())
    return bytes(out), shape, n


def dequantize_q8_0(data: bytes, n: int) -> np.ndarray:
    out = np.empty(((len(data) // 34) * 32,), dtype=np.float32)
    for b in range(len(data) // 34):
        chunk = data[b * 34 : (b + 1) * 34]
        scale = struct.unpack("<e", chunk[:2])[0]
        out[b * 32 : (b + 1) * 32] = np.frombuffer(chunk[2:], dtype=np.int8).astype(np.float32) * scale
    return out[:n]


def dequantize_q4_0(data: bytes, n: int) -> np.ndarray:
    out = np.empty(((len(data) // 18) * 32,), dtype=np.float32)
    for b in range(len(data) // 18):
        chunk = data[b * 18 : (b + 1) * 18]
        scale = struct.unpack("<e", chunk[:2])[0]
        packed = np.frombuffer(chunk[2:], dtype=np.uint8)
        low = (packed & 0x0F).astype(np.int8) - 8
        high = (packed >> 4).astype(np.int8) - 8
        out[b * 32 : b * 32 + 16] = low.astype(np.float32) * scale
        out[b * 32 + 16 : (b + 1) * 32] = high.astype(np.float32) * scale
    return out[:n]


def _write_string(f, s: str) -> None:
    enc = s.encode("utf-8")
    f.write(struct.pack("<Q", len(enc)))
    f.write(enc)


def _write_kv(f, key: str, vtype: int, value: Any) -> None:
    _write_string(f, key)
    f.write(struct.pack("<I", vtype))
    if vtype == GGUF_TYPE_UINT32:
        f.write(struct.pack("<I", int(value)))
    elif vtype == GGUF_TYPE_INT32:
        f.write(struct.pack("<i", int(value)))
    elif vtype == GGUF_TYPE_FLOAT32:
        f.write(struct.pack("<f", float(value)))
    elif vtype == GGUF_TYPE_BOOL:
        f.write(struct.pack("<?", bool(value)))
    elif vtype == GGUF_TYPE_STRING:
        _write_string(f, str(value))
    elif vtype == GGUF_TYPE_UINT64:
        f.write(struct.pack("<Q", int(value)))
    else:
        raise ExportError(f"unsupported GGUF metadata type {vtype}")


def export_gguf(state_dict: Mapping[str, torch.Tensor], spec: ModelSpec, output_path: str | Path,
                quantize: str = "none", model_name: str = "forgeline") -> Dict[str, Any]:
    """Write ``state_dict`` as a GGUF file. Returns a summary dict."""
    if quantize not in QUANT_FORMATS:
        raise ExportError(f"quantize must be one of {QUANT_FORMATS}, got {quantize!r}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    infos: List[Dict[str, Any]] = []
    for name, param in state_dict.items():
        if "mtp_modules" in name:
            continue
        tensor = param.detach().cpu()
        if quantize in ("q8_0", "q4_0") and tensor.dim() >= 2:
            data, shape, n = (quantize_q8_0 if quantize == "q8_0" else quantize_q4_0)(tensor)
            ttype = GGML_TYPE_Q8_0 if quantize == "q8_0" else GGML_TYPE_Q4_0
        elif tensor.dim() >= 2:
            data, shape, n, ttype = tensor.half().numpy().tobytes(), tuple(tensor.shape), tensor.numel(), GGML_TYPE_F16
        else:
            data, shape, n, ttype = tensor.float().numpy().tobytes(), tuple(tensor.shape), tensor.numel(), GGML_TYPE_F32
        infos.append({"name": name, "shape": list(shape), "type": ttype, "data": data, "n_elem": n})

    # compute aligned offsets within the data section
    offset = 0
    for info in infos:
        info["offset"] = offset
        offset += len(info["data"])
        offset += (GGUF_ALIGNMENT - offset % GGUF_ALIGNMENT) % GGUF_ALIGNMENT

    file_type = FILE_TYPE.get(GGML_TYPE_Q8_0 if quantize == "q8_0" else GGML_TYPE_Q4_0 if quantize == "q4_0" else GGML_TYPE_F16, 0)
    metadata = {
        "general.architecture": (GGUF_TYPE_STRING, "forgeline"),
        "general.name": (GGUF_TYPE_STRING, f"{model_name} {spec.n_layer}L"),
        "general.file_type": (GGUF_TYPE_UINT32, file_type),
        "general.alignment": (GGUF_TYPE_UINT32, GGUF_ALIGNMENT),
        "forgeline.context_length": (GGUF_TYPE_UINT32, spec.block_size),
        "forgeline.embedding_length": (GGUF_TYPE_UINT32, spec.n_embd),
        "forgeline.block_count": (GGUF_TYPE_UINT32, spec.n_layer),
        "forgeline.head_count": (GGUF_TYPE_UINT32, spec.n_head),
        "forgeline.head_count_kv": (GGUF_TYPE_UINT32, spec.n_kv_head),
        "forgeline.vocab_size": (GGUF_TYPE_UINT32, spec.vocab_size),
        "forgeline.use_moe": (GGUF_TYPE_BOOL, spec.use_moe),
        "forgeline.use_mla": (GGUF_TYPE_BOOL, spec.use_mla),
        "forgeline.use_swiglu": (GGUF_TYPE_BOOL, spec.use_swiglu),
        "forgeline.use_rope": (GGUF_TYPE_BOOL, spec.use_rope),
        "forgeline.sliding_window": (GGUF_TYPE_UINT32, spec.sliding_window),
    }

    with output_path.open("wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(infos)))
        f.write(struct.pack("<Q", len(metadata)))
        for key, (vtype, value) in metadata.items():
            _write_kv(f, key, vtype, value)
        for info in infos:
            _write_string(f, info["name"])
            f.write(struct.pack("<I", len(info["shape"])))
            for dim in reversed(info["shape"]):  # ggml stores dims innermost-first
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<I", info["type"]))
            f.write(struct.pack("<Q", info["offset"]))
        pos = f.tell()
        f.write(b"\x00" * ((GGUF_ALIGNMENT - pos % GGUF_ALIGNMENT) % GGUF_ALIGNMENT))
        data_start = f.tell()
        for info in infos:
            assert f.tell() - data_start == info["offset"]
            f.write(info["data"])
            f.write(b"\x00" * ((GGUF_ALIGNMENT - len(info["data"]) % GGUF_ALIGNMENT) % GGUF_ALIGNMENT))

    size = output_path.stat().st_size
    fp32_size = sum(int(p.numel()) * 4 for p in state_dict.values())
    return {
        "path": str(output_path), "bytes": size, "tensors": len(infos), "format": quantize,
        "fp32_bytes": fp32_size, "compression": size / max(fp32_size, 1),
    }


def read_gguf_header(path: str | Path) -> Dict[str, Any]:
    """Parse the header (metadata + tensor infos) of a GGUF file written by :func:`export_gguf`."""
    path = Path(path)
    with path.open("rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
        if magic != GGUF_MAGIC:
            raise ExportError(f"{path} is not a GGUF file")
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        def read_string() -> str:
            (length,) = struct.unpack("<Q", f.read(8))
            return f.read(length).decode("utf-8")

        metadata: Dict[str, Any] = {}
        for _ in range(n_kv):
            key = read_string()
            (vtype,) = struct.unpack("<I", f.read(4))
            if vtype == GGUF_TYPE_UINT32:
                metadata[key] = struct.unpack("<I", f.read(4))[0]
            elif vtype == GGUF_TYPE_INT32:
                metadata[key] = struct.unpack("<i", f.read(4))[0]
            elif vtype == GGUF_TYPE_FLOAT32:
                metadata[key] = struct.unpack("<f", f.read(4))[0]
            elif vtype == GGUF_TYPE_BOOL:
                metadata[key] = struct.unpack("<?", f.read(1))[0]
            elif vtype == GGUF_TYPE_STRING:
                metadata[key] = read_string()
            elif vtype == GGUF_TYPE_UINT64:
                metadata[key] = struct.unpack("<Q", f.read(8))[0]
            else:
                raise ExportError(f"unsupported metadata type {vtype} in {path}")
        tensors = []
        for _ in range(n_tensors):
            name = read_string()
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ttype, offset = struct.unpack("<IQ", f.read(12))
            tensors.append({"name": name, "shape": list(reversed(dims)), "type": ttype, "offset": offset})
        pos = f.tell()
        data_start = pos + (GGUF_ALIGNMENT - pos % GGUF_ALIGNMENT) % GGUF_ALIGNMENT
    return {"version": version, "metadata": metadata, "tensors": tensors, "data_start": data_start}


def read_gguf_tensor(path: str | Path, name: str) -> np.ndarray:
    header = read_gguf_header(path)
    info = next((t for t in header["tensors"] if t["name"] == name), None)
    if info is None:
        raise ExportError(f"tensor {name!r} not found in {path}")
    n = int(np.prod(info["shape"])) if info["shape"] else 1
    with Path(path).open("rb") as f:
        f.seek(header["data_start"] + info["offset"])
        if info["type"] == GGML_TYPE_F32:
            arr = np.frombuffer(f.read(n * 4), dtype=np.float32)
        elif info["type"] == GGML_TYPE_F16:
            arr = np.frombuffer(f.read(n * 2), dtype=np.float16).astype(np.float32)
        elif info["type"] == GGML_TYPE_Q8_0:
            n_blocks = (n + 31) // 32
            arr = dequantize_q8_0(f.read(n_blocks * 34), n)
        elif info["type"] == GGML_TYPE_Q4_0:
            n_blocks = (n + 31) // 32
            arr = dequantize_q4_0(f.read(n_blocks * 18), n)
        else:
            raise ExportError(f"unsupported tensor type {info['type']}")
    return arr.reshape(info["shape"])
