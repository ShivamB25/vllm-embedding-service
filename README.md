# vLLM Embedding Service on Modal

Production deployment of **tencent/KaLM-Embedding-Gemma3-12B-2511** (12B params) with ultra-fast GPU snapshots.

## 🚀 Quick Deploy

```bash
pip install modal openai
modal token new

# Option 1: Standard HTTP Server (OpenAI-compatible) - ~30s cold start
modal deploy modal_vllm_embedding.py

# Option 2: GPU Snapshot (RECOMMENDED) - 5-10s cold start ⚡
modal deploy modal_vllm_embedding_with_snapshot.py
```

## 📖 Complete Documentation

**👉 See [`VLLM_COMPLETE_GUIDE.md`](./VLLM_COMPLETE_GUIDE.md)** for everything:

- ✅ Detailed setup & deployment
- ✅ Both HTTP and GPU snapshot options
- ✅ Usage examples (OpenAI, LangChain, LlamaIndex)
- ✅ GPU snapshots explained (3-6x faster cold starts!)
- ✅ Performance benchmarks
- ✅ Cost optimization (save up to 70%)
- ✅ Troubleshooting guide
- ✅ Integration examples

## 🎯 Performance

| Option | Cold Start | Access | Best For |
|--------|-----------|---------|----------|
| Standard HTTP | ~30s | OpenAI API | REST APIs, integrations |
| **GPU Snapshot** ⚡ | **5-10s** | Python API | Python apps, cost optimization |

## 📁 Files

- `modal_vllm_embedding.py` - Standard HTTP server
- `modal_vllm_embedding_with_snapshot.py` - GPU snapshot version (RECOMMENDED)
- `test_embedding_client.py` - Test suite
- `VLLM_COMPLETE_GUIDE.md` - Full documentation
- `README.md` - This file

## 💡 Quick Example

```python
# HTTP Server
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url="https://your-app.modal.run/v1")
response = client.embeddings.create(input=["Hello"], model="tencent/KaLM-Embedding-Gemma3-12B-2511")

# GPU Snapshot (faster!)
from modal import Cls
VLLMEmbedding = Cls.from_name("vllm-embedding-snapshot", "VLLMEmbeddingSnapshot")
embeddings = VLLMEmbedding().embed.remote(["Hello world"])
```

## 🔗 Built With

- Context7 (`/modal-labs/modal-examples`, `/websites/modal`, `/websites/vllm_ai_en`)
- Sequential Thinking for planning
- Modal for serverless GPU infrastructure
- vLLM for optimized inference

**Read the complete guide:** [`VLLM_COMPLETE_GUIDE.md`](./VLLM_COMPLETE_GUIDE.md)
