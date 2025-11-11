"""
Example client for testing the deployed vLLM embedding service
Replace BASE_URL with your Modal deployment URL
"""

from openai import OpenAI

# Replace with your Modal deployment URL after running `modal deploy`
BASE_URL = "https://your-workspace--vllm-embedding-service-serve.modal.run/v1"


def main():
    # Initialize OpenAI client pointing to vLLM server
    client = OpenAI(api_key="EMPTY", base_url=BASE_URL)  # vLLM doesn't require authentication

    print("=" * 60)
    print("vLLM Embedding Service Test")
    print("=" * 60)

    # Test 1: Single embedding
    print("\n[Test 1] Single embedding:")
    response = client.embeddings.create(
        input="This is a test sentence for embedding generation.",
        model="tencent/KaLM-Embedding-Gemma3-12B-2511",
    )

    embedding = response.data[0].embedding
    print(f"✓ Embedding dimension: {len(embedding)}")
    print(f"✓ First 10 values: {embedding[:10]}")
    print(f"✓ Usage: {response.usage.prompt_tokens} tokens")

    # Test 2: Batch embeddings
    print("\n[Test 2] Batch embeddings:")
    sentences = [
        "The cat sat on the mat.",
        "Machine learning is transforming technology.",
        "Modal makes it easy to deploy AI models.",
        "vLLM provides fast inference for LLMs.",
    ]

    response = client.embeddings.create(
        input=sentences, model="tencent/KaLM-Embedding-Gemma3-12B-2511"
    )

    print(f"✓ Generated {len(response.data)} embeddings")
    for i, data in enumerate(response.data):
        print(f"  - Sentence {i + 1}: {len(data.embedding)} dimensions")

    # Test 3: Semantic similarity
    print("\n[Test 3] Semantic similarity:")
    import numpy as np

    test_queries = [
        "A dog playing in the park",
        "Machine learning algorithms",
    ]

    response = client.embeddings.create(
        input=test_queries, model="tencent/KaLM-Embedding-Gemma3-12B-2511"
    )

    emb1 = np.array(response.data[0].embedding)
    emb2 = np.array(response.data[1].embedding)

    # Cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"✓ Cosine similarity between queries: {similarity:.4f}")

    # Test 4: Performance test
    print("\n[Test 4] Performance test:")
    import time

    test_sentence = "Performance testing embedding generation speed."
    num_requests = 5

    start = time.time()
    for _ in range(num_requests):
        client.embeddings.create(
            input=test_sentence, model="tencent/KaLM-Embedding-Gemma3-12B-2511"
        )
    end = time.time()

    avg_latency = (end - start) / num_requests * 1000  # ms
    print(f"✓ Average latency: {avg_latency:.2f}ms per request")
    print(f"✓ Total time: {(end - start):.2f}s for {num_requests} requests")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. The service is deployed: modal deploy modal_vllm_embedding.py")
        print("2. BASE_URL is set to your Modal deployment URL")
        print("3. numpy and openai are installed: pip install openai numpy")
