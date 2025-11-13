# vLLM Embedding Service on Modal - Complete Guide

Production deployment of **tencent/KaLM-Embedding-Gemma3-12B-2511** using Modal's GPU memory snapshot workflow. This guide focuses on the `VLLMEmbeddingSnapshot` class in `modal_vllm_embedding_with_snapshot.py`, which is the supported path forward.

---

## Contents

1. [Quick Start](#quick-start)
2. [Deployment Workflow](#deployment-workflow)
3. [Usage Patterns](#usage-patterns)
4. [Configuration Reference](#configuration-reference)
5. [GPU Snapshot Details](#gpu-snapshot-details)
6. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
7. [Cost & Scaling Guidance](#cost--scaling-guidance)
8. [References](#references)

---

## Quick Start

### Prerequisites

```bash
pip install modal openai numpy
modal token new
```

### Deploy the application

1. Pre-warm the Hugging Face cache (optional but avoids large downloads during deploy):

   ```bash
   modal run modal_vllm_embedding_with_snapshot.py::download_model_weights
   ```

2. Deploy the GPU snapshot application:

   ```bash
   modal deploy modal_vllm_embedding_with_snapshot.py
   ```

3. Invoke the class from any Modal-capable environment:

   ```python
   from modal import Cls

   VLLMEmbedding = Cls.from_name("vllm-embedding-snapshot", "VLLMEmbeddingSnapshot")
   result = VLLMEmbedding().embed.remote(["Hello world"])
   ```

The legacy standalone `modal_vllm_embedding.py` module has been removed. If you still need an OpenAI-compatible HTTP endpoint, use the `serve_http()` function defined in the same file; it runs without GPU snapshots and cold starts in ~30 seconds.

---

## Deployment Workflow

1. **Image build** – `modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")` installs vLLM 0.10.2, PyTorch 2.8.0, and Hugging Face transfer acceleration.
2. **Volumes** – `hf-embedding-cache` stores model weights and `vllm-jit-cache` stores JIT artifacts. Both volumes are mounted into `/root/.cache`. Commit volumes after downloads to persist updates.
3. **`download_model_weights`** – optional function that downloads weights once per change so deployment can immediately snapshot.
4. **`VLLMEmbeddingSnapshot` class** – single-container Modal class with `enable_memory_snapshot=True` and `experimental_options={"enable_gpu_snapshot": True}`. `@modal.enter(snap=True)` loads the model onto GPU before snapshotting. Cold starts restore in 5–10 seconds.
5. **Methods** – `embed` returns raw embeddings; `embed_with_metadata` returns dictionaries containing text, embedding, and dimension metadata, which is useful for Matryoshka Representation Learning (MRL) scenarios.
6. **Optional HTTP server** – `serve_http()` wraps `vllm serve` in a subprocess and exposes `/v1/embeddings`. Use this only when OpenAI-compatible HTTP access is required and longer cold starts are acceptable.

---

## Usage Patterns

### Basic call

```python
from modal import Cls

sentences = ["The GPU snapshot is online", "Modal keeps weights cached"]
VLLMEmbedding = Cls.from_name("vllm-embedding-snapshot", "VLLMEmbeddingSnapshot")
embeddings = VLLMEmbedding().embed.remote(sentences)

for emb in embeddings:
    assert len(emb) == 3840
```

### Metadata output

```python
with_metadata = VLLMEmbedding().embed_with_metadata.remote(["Short text"])  # [{'text': ..., 'embedding': [...], 'dimension': 3840}]
```

### Async fan-out

```python
import asyncio

async def run_batches(queries):
    VLLMEmbedding = Cls.from_name("vllm-embedding-snapshot", "VLLMEmbeddingSnapshot")
    coros = [VLLMEmbedding().embed.remote.aio(batch) for batch in queries]
    return await asyncio.gather(*coros)
```

When batching, stay under the recommended 512-token sequence length unless you understand the throughput trade-offs up to the 32,000-token hard limit.

---

## Configuration Reference

### Model facts

- Model: `tencent/KaLM-Embedding-Gemma3-12B-2511` (11.76B parameters, BF16)
- Embedding dimension: 3,840 by default with MRL support for 3,840 / 2,048 / 1,024 / 512 / 256 / 128 / 64
- Pooling: last-token pooling as implemented by vLLM embed task
- Max tokens: 32,000 (512 recommended for latency/quality balance)
- Hardware: A100-40GB GPU, `gpu_memory_utilization=0.9`, `max_num_seqs=256`
- Performance baseline: 5–10 s cold start, ~50–100 ms warm latency on single requests

### Key parameters

| Setting | Value | Notes |
|--------|-------|-------|
| `task` | `embed` | Ensures vLLM runs in embedding mode |
| `enforce_eager` | `True` | Required when GPU snapshots are enabled |
| `revision` | `"CausalLM"` | Gemma-3 embeddings require the CausalLM revision |
| `device` | `"cuda"` | Must load directly to GPU during snapshot capture |
| `dtype` | `"auto"` | vLLM selects BF16 based on model config |
| `max_num_seqs` | `256` | Good balance for concurrent requests |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Speeds up weight downloads |
| Volumes | `hf-embedding-cache`, `vllm-jit-cache` | Avoid repeat downloads and recompiles |
| Scaling | `max_containers=1`, `scaledown_window=5*MINUTES` | Keeps the deployment single-tenant and scales to zero quickly |

### Optional HTTP endpoint

Running `modal run modal_vllm_embedding_with_snapshot.py::serve_http` exposes `/v1/embeddings`, `/health`, and `/v1/models` via the vLLM OpenAI-compatible server. This path cannot use GPU memory snapshots, so plan for ~30-second cold starts.

---

## GPU Snapshot Details

```python
@app.cls(
    gpu="A100-40GB",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class VLLMEmbeddingSnapshot:
    @modal.enter(snap=True)
    def load_model(self):
        from vllm import LLM

        self.model = LLM(
            model=MODEL_NAME,
            task="embed",
            enforce_eager=True,
            device="cuda",
            dtype="auto",
            gpu_memory_utilization=0.9,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            max_num_seqs=256,
        )

    @modal.method()
    def embed(self, sentences: list[str]):
        outputs = self.model.embed(sentences)
        return [output.outputs.embedding for output in outputs]
```

Snapshot creation happens on the first deployment after code changes. Expect roughly two minutes for the initial `modal deploy` so that Modal can capture GPU state. Later cold starts simply restore the snapshot, so you can keep `scaledown_window` short without paying the 30-second penalty.

---

## Monitoring & Troubleshooting

- List apps: `modal app list`
- Inspect the deployment: `modal app show vllm-embedding-snapshot`
- Tail logs: `modal app logs vllm-embedding-snapshot --follow`
- Monitor containers: `modal container list --app vllm-embedding-snapshot`
- Verify snapshot creation: `modal app logs vllm-embedding-snapshot | rg snapshot`

### Measuring cold start

```python
import time
from modal import Cls

VLLMEmbedding = Cls.from_name("vllm-embedding-snapshot", "VLLMEmbeddingSnapshot")
start = time.time()
_ = VLLMEmbedding().embed.remote(["ping"])
print(f"Elapsed: {time.time() - start:.2f}s")
```

### Common issues

| Symptom | Mitigation |
|---------|------------|
| OOM during load | Lower `gpu_memory_utilization` to 0.85 or select A100-80GB |
| Snapshot never restores | Make sure `modal deploy` ran successfully and that `@modal.enter(snap=True)` finished without raising |
| Slow downloads | Confirm `HF_HUB_ENABLE_HF_TRANSFER=1` and run `download_model_weights` once per change |
| Idle GPU cost | Reduce `scaledown_window` (default 5 minutes) and rely on GPU snapshot recovery |

---

## Cost & Scaling Guidance

- **Scale to zero** – With GPU snapshots you can keep `max_containers=1` and aggressively scale down after five idle minutes without sacrificing responsiveness.
- **Batch requests** – Prefer batches of 16–64 sentences so vLLM can reuse cached KV data and reduce per-request overhead.
- **Use A100-40GB** – The model fits comfortably in 40 GB; moving to 80 GB adds cost without benefits.
- **Cache once** – Keep both volumes attached so weights and vLLM JIT artifacts persist across deployments.

Approximate GPU pricing on Modal (subject to change): A10G ≈ $0.006/min, A100-40GB ≈ $0.016/min, A100-80GB ≈ $0.023/min, H100 ≈ $0.034/min. This service is optimized for A100-40GB.

---

## References

- Modal documentation: https://modal.com/docs/
- Modal GPU memory snapshots: https://modal.com/docs/guide/memory-snapshot
- Modal examples (vLLM): https://github.com/modal-labs/modal-examples
- vLLM documentation: https://docs.vllm.ai/
- Model card: https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511

---

For historical notes on the removed standard deployment, consult git history. All new work should target the GPU snapshot pathway documented here.
