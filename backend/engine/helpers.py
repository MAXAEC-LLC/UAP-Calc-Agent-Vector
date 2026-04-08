"""
helpers.py — Core utility functions for the Multi-Agent System.
Implements: hardened LLM calls, embeddings, Pinecone queries, MCP messaging,
token counting, input sanitization, and content moderation.
"""

import logging
import re
import json
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# === Hardened LLM Call with Retry ===

@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APITimeoutError, anthropic.APIConnectionError)),
    before_sleep=lambda retry_state: logging.warning(
        f"LLM call failed (attempt {retry_state.attempt_number}). Retrying..."
    ),
)
def call_llm_robust(system_prompt, user_prompt, client, generation_model, json_mode=False, temperature=0.1):
    """Makes a robust, retrying call to the Anthropic Messages API."""
    kwargs = {
        "model": generation_model,
        "max_tokens": 16384,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    response = client.messages.create(**kwargs)
    text = response.content[0].text

    if json_mode:
        # Strip markdown code fences that Claude may wrap JSON in
        stripped = text.strip()
        if stripped.startswith("```"):
            first_newline = stripped.index("\n") if "\n" in stripped else len(stripped)
            stripped = stripped[first_newline + 1:]
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()[:-3].rstrip()
            text = stripped

    return text


def call_llm_stream(system_prompt, user_prompt, client, generation_model, temperature=0.1):
    """Makes a streaming call to the Anthropic Messages API. Yields text chunks."""
    with client.messages.stream(
        model=generation_model,
        max_tokens=16384,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    ) as stream:
        for text in stream.text_stream:
            yield text


# === Embedding Helper ===

def get_embedding(text, client, embedding_model):
    """Generates an embedding vector for a single text string."""
    response = client.embeddings.create(input=[text], model=embedding_model)
    return response.data[0].embedding


def get_embeddings_batch(texts, client, embedding_model, batch_size=128):
    """Embed a list of texts in batches, returning a list of embedding vectors."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(input=batch, model=embedding_model)
        all_embeddings.extend(d.embedding for d in response.data)
    return all_embeddings


# === Pinecone Query ===

def query_pinecone(query_text, index, client, embedding_model, namespace, top_k=10):
    """Embeds a query and searches a Pinecone namespace."""
    query_embedding = get_embedding(query_text, client, embedding_model)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )
    return results.get("matches", [])


# === MCP v2.0 Message Protocol ===

def create_mcp_message(sender, content, metadata=None):
    """Creates a standardized MCP v2.0 inter-agent message."""
    return {
        "sender": sender,
        "content": content,
        "metadata": metadata or {},
    }


# === Token Counting ===

def count_tokens(text, model="gpt-4"):
    """Counts the number of tokens in a text string using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(str(text)))


# === Input Sanitization ===

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now",
    r"disregard\s+(all\s+)?prior",
    r"system\s*:\s*",
    r"<\|im_start\|>",
    r"###\s*instruction",
    r"ADMIN\s*OVERRIDE",
]
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def helper_sanitize_input(text):
    """Checks user input for common prompt injection patterns."""
    for pattern in COMPILED_PATTERNS:
        if pattern.search(text):
            logging.warning(f"Prompt injection attempt detected: {pattern.pattern}")
            raise ValueError("Input rejected: potential prompt injection detected.")
    return text


# === Content Moderation ===

def helper_moderate_content(text, client):
    """Uses the OpenAI Moderation API to check content safety."""
    response = client.moderations.create(input=text)
    result = response.results[0]
    if result.flagged:
        flagged_categories = [
            cat for cat, flagged in result.categories.model_dump().items() if flagged
        ]
        logging.warning(f"Content flagged by moderation: {flagged_categories}")
        raise ValueError(f"Content flagged by moderation: {', '.join(flagged_categories)}")
    return text


logging.info("✅ Helpers module loaded.")
