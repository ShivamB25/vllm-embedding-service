# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation Sources

**IMPORTANT**: When updating code or documentation in this repository, STRICTLY use these official Modal resources for context and accuracy:

1. **Official Modal Documentation**: https://modal.com/docs/
   - Primary source for Modal API reference, features, and best practices
   - Use for GPU snapshots, volumes, deployment patterns, and configuration

2. **Modal Examples Repository**: https://github.com/modal-labs/modal-examples
   - Reference implementations and patterns
   - vLLM integration examples and embedding service patterns

**Do NOT rely on outdated information or make assumptions about Modal features without verifying against these sources.**

## Project Overview

This is a production vLLM embedding service deployment on Modal, serving the **tencent/KaLM-Embedding-Gemma3-12B-2511** model (11.76B parameters). The primary interface is the GPU snapshot-based Python API (5–10 s cold starts). An optional OpenAI-compatible HTTP server (~30 s cold starts) is still available via the `serve_http()` function inside `modal_vllm_embedding_with_snapshot.py`.

### Model Specifications

- **Parameters**: 11.76B (fine-tuned from Google's Gemma-3-12B)
- **Max Input Tokens**: 32,000
- **Recommended Sequence Length**: 512 tokens for typical usage
- **Embedding Dimension**: 3840
- **MRL Support**: Matryoshka Representation Learning with variable dimensions (3840, 2048, 1024, 512, 256, 128, 64)
- **Pooling Method**: Last token pooling
- **Data Type**: BF16 (bfloat16)
- **Performance**: Ranks #1 on MMTEB benchmarks (Nov 2025) with 72.32 mean score

## Development Commands

### Initial Setup
```bash
pip install modal openai numpy
modal token new
```

### Deployment
```bash
# Deploy the GPU snapshot app (recommended)
modal deploy modal_vllm_embedding_with_snapshot.py

# Optional: warm caches or iterate locally
modal run modal_vllm_embedding_with_snapshot.py
```

### Testing
```bash
# Exercise the HTTP endpoint (update BASE_URL first) – only relevant if serve_http() is in use
python test_embedding_client.py

# Local validation for the snapshot class
modal run modal_vllm_embedding_with_snapshot.py
```

## Architecture

### Deployment Surfaces

1. **GPU Snapshot API (Python)** – `VLLMEmbeddingSnapshot` class
   - Uses Modal's `enable_memory_snapshot=True` with `@modal.enter(snap=True)`
   - Model is loaded onto GPU during snapshot creation, then restored instantly
   - Cold start: 5–10 seconds (versus ~30 seconds without snapshots)
   - Access via Modal's `Cls.from_name()` / `.remote()` pattern
   - Default for production traffic; all new work should target this path

2. **HTTP Server (optional)** – `serve_http()` function in `modal_vllm_embedding_with_snapshot.py`
   - OpenAI-compatible `/v1/embeddings` endpoint implemented via `vllm serve`
   - Runs as a subprocess (no GPU snapshot support)
   - Cold start: ~30 seconds
   - Use only when REST API integrations or non-Python clients are unavoidable

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

Key vLLM parameters set in `modal_vllm_embedding_with_snapshot.py:96-106`:
- `task="embed"`: Embedding mode (not text generation)
- `enforce_eager=True`: Disables CUDA graph optimization for snapshot compatibility
- `gpu_memory_utilization=0.9`: Uses 90% of A100-40GB memory
- `revision="CausalLM"`: Required because vLLM only supports Gemma3ForCausalLM variant
- `trust_remote_code=True`: Required for custom model code
- `device="cuda"`: Load directly onto GPU for snapshot creation
- `dtype="auto"`: Auto-detects BF16 precision from model configuration
- `max_num_seqs=256`: Optimizes batch processing throughput for concurrent requests
- Maximum sequence length is auto-detected (32K max tokens, 512 recommended for typical usage)

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
- Timeout: 5 minutes across inference, HTTP endpoint, and the optional weight download helper
- Container lifecycle: 5-minute idle period before scale-down
- Autoscaling: `max_containers=1` on both `VLLMEmbeddingSnapshot` and `serve_http()` so the service never scales past a single GPU-backed container (per Modal scaling guide)

## Important Notes

- The snapshot version uses vLLM's Python API directly, not the CLI server
- HTTP server uses subprocess approach and cannot leverage GPU snapshots
- First deployment after code changes triggers a new snapshot creation (~2 minutes)
- The model revision "CausalLM" is required because vLLM only supports Gemma3ForCausalLM
- Both deployment modes share the same cached volumes for efficiency
- **MRL Support**: Model supports Matryoshka Representation Learning with variable dimensions (3840 down to 64). This allows flexibility in embedding size vs. performance trade-offs
- **Token Limits**: While the model supports up to 32,000 tokens, 512 tokens is recommended for typical usage to balance performance and accuracy
- **Embedding Dimension**: Outputs are 3840-dimensional vectors by default (can be reduced via MRL)
