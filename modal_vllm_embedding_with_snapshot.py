"""
Ultra-fast vLLM Embedding with GPU Memory Snapshots
Cold starts: 5-10 seconds (vs 30s without snapshots)

This version uses vLLM's Python API directly instead of the server,
enabling GPU memory snapshots for instant cold starts.
"""

import modal

# Constants
MODEL_NAME = "tencent/KaLM-Embedding-Gemma3-12B-2511"
MODEL_REVISION = "CausalLM"
APP_NAME = "vllm-embedding-snapshot"
MINUTES = 60

# Create Modal app
app = modal.App(APP_NAME)

# Volumes for caching
hf_cache_vol = modal.Volume.from_name("hf-embedding-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-jit-cache", create_if_missing=True)

# Configure image with vLLM - use uv for faster installs
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install(
        "vllm==0.10.2",
        "huggingface_hub[hf_transfer]==0.35.0",
        "torch==2.8.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": "/root/.cache/huggingface",
        }
    )
)


# Pre-download weights
@app.function(
    image=vllm_image,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    timeout=20 * MINUTES,
)
def download_model_weights():
    """Download model weights to cache volume"""
    from huggingface_hub import snapshot_download

    print(f"Downloading model: {MODEL_NAME}")
    snapshot_download(
        MODEL_NAME,
        revision=MODEL_REVISION,
        ignore_patterns=["*.pt", "*.bin"],
    )
    hf_cache_vol.commit()
    print("Model weights cached!")


@app.cls(
    image=vllm_image,
    gpu="A100-40GB",
    scaledown_window=5 * MINUTES,  # Combined: scale down after 5 mins idle
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    # Enable GPU memory snapshots for ultra-fast cold starts
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class VLLMEmbeddingSnapshot:
    """
    Ultra-fast embedding service using GPU memory snapshots
    Cold start: 5-10 seconds (vs 30s without snapshots)
    """

    @modal.enter(snap=True)  # This runs during snapshot creation
    def load_model(self):
        """
        Load vLLM model onto GPU - this state is captured in the snapshot!
        Next time the container starts, it loads directly from GPU snapshot.
        """
        from vllm import LLM

        print(f"Loading model {MODEL_NAME} onto GPU...")
        self.model = LLM(
            model=MODEL_NAME,
            task="embed",
            enforce_eager=True,
            revision=MODEL_REVISION,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            device="cuda",  # Load directly onto GPU for snapshot
        )
        print("Model loaded onto GPU - ready for snapshot!")
        vllm_cache_vol.commit()

    @modal.method()
    def embed(self, sentences: list[str]) -> list[list[float]]:
        """
        Generate embeddings (restored from snapshot = instant!)

        Args:
            sentences: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        print(f"Generating embeddings for {len(sentences)} sentences")
        outputs = self.model.embed(sentences)
        embeddings = [output.outputs.embedding for output in outputs]
        return embeddings

    @modal.method()
    def embed_with_metadata(self, sentences: list[str]) -> list[dict[str, any]]:
        """
        Generate embeddings with metadata

        Returns:
            List of dicts with 'text', 'embedding', and 'dimension'
        """
        outputs = self.model.embed(sentences)
        results = []

        for i, output in enumerate(outputs):
            embedding = output.outputs.embedding
            results.append(
                {
                    "text": sentences[i],
                    "embedding": embedding,
                    "dimension": len(embedding),
                }
            )

        return results


# Also provide OpenAI-compatible HTTP endpoint (without snapshots)
# This uses the subprocess approach which can't use GPU snapshots
@app.function(
    image=vllm_image,
    gpu="A100-40GB",
    scaledown_window=5 * MINUTES,  # Scale down after 5 mins idle
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.web_server(port=8000, startup_timeout=10 * MINUTES)
def serve_http():
    """
    OpenAI-compatible HTTP server (standard cold start ~30s)
    Use this for HTTP/REST access
    Use VLLMEmbeddingSnapshot class for fastest Python API access
    """
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--task",
        "embed",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.9",
        "--trust-remote-code",
        "--served-model-name",
        MODEL_NAME,
    ]

    print(f"Starting vLLM HTTP server: {' '.join(cmd)}")
    subprocess.Popen(cmd)


@app.local_entrypoint()
def main():
    """Test both approaches"""

    # Step 1: Download weights
    print("=" * 60)
    print("Step 1: Downloading model weights...")
    print("=" * 60)
    download_model_weights.remote()

    # Step 2: Test snapshot-enabled Python API (FASTEST)
    print("\n" + "=" * 60)
    print("Step 2: Testing GPU Snapshot API (5-10s cold start)")
    print("=" * 60)

    embedder = VLLMEmbeddingSnapshot()

    test_sentences = [
        "This is a test sentence",
        "GPU snapshots make cold starts instant!",
        "Modal's GPU snapshots are amazing",
    ]

    print("\n[Python API Test]")
    embeddings = embedder.embed.remote(test_sentences)
    print(f"✓ Generated {len(embeddings)} embeddings")
    print(f"✓ Embedding dimension: {len(embeddings[0])}")

    # Test with metadata
    print("\n[Metadata Test]")
    results = embedder.embed_with_metadata.remote(test_sentences[:2])
    for result in results:
        print(f"✓ Text: {result['text'][:50]}...")
        print(f"  Dimension: {result['dimension']}")

    print("\n" + "=" * 60)
    print("DEPLOYMENT INSTRUCTIONS")
    print("=" * 60)
    print(
        """
For GPU Snapshot API (Python, fastest - 5-10s cold starts):
1. Deploy: modal deploy modal_vllm_embedding_with_snapshot.py
2. First call creates the snapshot (takes ~2 mins)
3. Subsequent calls restore from snapshot (5-10s!)

Usage:
from modal import Cls
VLLMEmbedding = Cls.from_name("vllm-embedding-snapshot", "VLLMEmbeddingSnapshot")
embeddings = VLLMEmbedding().embed.remote(["Hello world"])

For HTTP API (OpenAI-compatible, ~30s cold starts):
- Same URL pattern as before
- Use OpenAI client as shown in other examples
    """
    )


if __name__ == "__main__":
    main()
