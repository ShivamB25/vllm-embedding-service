# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a production vLLM embedding service deployment on Modal, serving the **tencent/KaLM-Embedding-Gemma3-12B-2511** model (12B parameters). The service provides two deployment options: a GPU snapshot-based Python API (5-10s cold starts) and an OpenAI-compatible HTTP server (~30s cold starts).

## Development Commands

### Initial Setup
```bash
pip install modal openai numpy
modal token new
```

### Deployment
```bash
# GPU Snapshot version (RECOMMENDED - 5-10s cold start)
modal deploy modal_vllm_embedding_with_snapshot.py

# Test the deployment
modal run modal_vllm_embedding_with_snapshot.py
```

### Testing
```bash
# Test the HTTP endpoint (update BASE_URL in file first)
python test_embedding_client.py

# Local testing with Modal
modal run modal_vllm_embedding_with_snapshot.py
```

## Architecture

### Two-Tier Deployment Strategy

1. **GPU Snapshot API (Python)** - `VLLMEmbeddingSnapshot` class
   - Uses Modal's `enable_memory_snapshot=True` with `@modal.enter(snap=True)`
   - Model is loaded onto GPU during snapshot creation, then restored instantly
   - Cold start: 5-10 seconds (vs 30s without snapshots)
   - Access via Modal's `Cls.from_name()` and `.remote()` pattern
   - Best for Python applications and cost optimization

2. **HTTP Server** - `serve_http()` function
   - OpenAI-compatible `/v1/embeddings` endpoint
   - Uses vLLM subprocess server (cannot use GPU snapshots)
   - Cold start: ~30 seconds
   - Best for REST API integrations and non-Python clients

### GPU Snapshot Mechanism

The snapshot workflow is critical to understand:
- `@modal.enter(snap=True)` marks the `load_model()` method for snapshot capture
- First deployment creates a GPU memory snapshot after loading the model onto CUDA
- Subsequent cold starts restore directly from GPU snapshot (3-6x faster)
- The `experimental_options={"enable_gpu_snapshot": True}` enables this feature

### Volume Strategy

Two persistent volumes are used:
- `hf-embedding-cache`: HuggingFace model weights cache
- `vllm-jit-cache`: vLLM JIT compilation cache
Both are mounted to avoid re-downloading/recompiling on each cold start.

### Model Configuration

Key vLLM parameters set in `modal_vllm_embedding_with_snapshot.py:90-100`:
- `task="embed"`: Embedding mode (not text generation)
- `enforce_eager=True`: Disables CUDA graph optimization for snapshot compatibility
- `gpu_memory_utilization=0.9`: Uses 90% of A100-40GB memory
- `max_model_len=8192`: Maximum sequence length
- `revision=MODEL_REVISION`: Set to "CausalLM" for this model
- `trust_remote_code=True`: Required for custom model code

## Usage Patterns

### Python API (Snapshot-based)
```python
from modal import Cls
VLLMEmbedding = Cls.from_name("vllm-embedding-snapshot", "VLLMEmbeddingSnapshot")
embeddings = VLLMEmbedding().embed.remote(["Hello world"])
```

### HTTP API (OpenAI-compatible)
```python
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url="https://your-app.modal.run/v1")
response = client.embeddings.create(input=["Hello"], model="tencent/KaLM-Embedding-Gemma3-12B-2511")
```

## Key Implementation Details

### Cold Start Optimization
The service uses multiple strategies to minimize cold starts:
1. GPU memory snapshots (primary optimization)
2. HuggingFace transfer acceleration (`HF_HUB_ENABLE_HF_TRANSFER=1`)
3. Volume-based caching for model weights and JIT artifacts
4. `scaledown_window=5*MINUTES` to keep containers warm during bursts

### Model Methods
- `embed(sentences: list[str])`: Returns list of embedding vectors
- `embed_with_metadata(sentences: list[str])`: Returns dicts with text, embedding, and dimension info

### Resource Allocation
- GPU: A100-40GB (required for 12B model)
- Timeout: 10 minutes for inference, 20 minutes for weight download
- Container lifecycle: 5-minute idle period before scale-down

## Important Notes

- The snapshot version uses vLLM's Python API directly, not the CLI server
- HTTP server uses subprocess approach and cannot leverage GPU snapshots
- First deployment after code changes triggers a new snapshot creation (~2 minutes)
- The model revision "CausalLM" is specific to this model variant
- Both deployment modes share the same cached volumes for efficiency
