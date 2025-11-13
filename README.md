# vLLM Embedding Service on Modal

This repository contains the production deployment of **tencent/KaLM-Embedding-Gemma3-12B-2511** on Modal. The only supported interface is the GPU snapshot powered `VLLMEmbeddingSnapshot` class defined in `modal_vllm_embedding_with_snapshot.py`.

## Quick Start

```bash
pip install modal openai numpy
modal token new

# Deploy the GPU snapshot application (5–10 s cold starts)
modal deploy modal_vllm_embedding_with_snapshot.py

# Run it locally during development
modal run modal_vllm_embedding_with_snapshot.py
```

## Usage

```python
from modal import Cls

VLLMEmbedding = Cls.from_name("vllm-embedding-snapshot", "VLLMEmbeddingSnapshot")
embeddings = VLLMEmbedding().embed.remote(["Hello world"])
print(len(embeddings[0]))  # 3840 dimensions
```

The class also exposes `embed_with_metadata` for callers that need Matryoshka Representation Learning (MRL) metadata or alternative embedding dimensions.

## Model & Deployment Facts

- Model: `tencent/KaLM-Embedding-Gemma3-12B-2511` (11.76B parameters, BF16)
- Max input tokens: 32,000 (512 recommended for most workloads)
- Embedding dimension: 3,840 with MRL support for 3,840 → 64
- Pooling method: last-token pooling via vLLM embed task
- Hardware: A100-40GB GPU, `gpu_memory_utilization=0.9`, `max_num_seqs=256`
- Persistent volumes: `hf-embedding-cache` and `vllm-jit-cache`
- Cold start strategy: Modal GPU memory snapshot using `@modal.enter(snap=True)` and `experimental_options={"enable_gpu_snapshot": True}`

## Files

- `modal_vllm_embedding_with_snapshot.py` – Modal app containing `VLLMEmbeddingSnapshot`
- `test_embedding_client.py` – Simple OpenAI-compatible client for smoke tests (update `BASE_URL` before running)
- `VLLM_COMPLETE_GUIDE.md` – Extended documentation for this deployment

## Additional Resources

Refer to [`VLLM_COMPLETE_GUIDE.md`](./VLLM_COMPLETE_GUIDE.md) for snapshot internals, configuration guidance, troubleshooting, and integration patterns.
