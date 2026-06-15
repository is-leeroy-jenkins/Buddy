# Embeddings

Embeddings mode creates vector representations of text for semantic comparison and retrieval.

## 🧭 Purpose

Embeddings support semantic search, document retrieval, similarity scoring, and retrieval-augmented
generation. Buddy uses embeddings to represent text chunks numerically so related material can be
found even when the wording does not match exactly.

## 🧱 Workflow Position

Embeddings sit between document preparation and retrieval. Source text is normalized, chunked,
embedded, stored, and later compared against a query vector.

| Stage | Description |
| --- | --- |
| Normalize | Cleans text for consistent comparison. |
| Chunk | Splits text into manageable segments. |
| Embed | Converts each chunk into a vector. |
| Store | Persists vectors in local or provider storage. |
| Search | Compares query vectors against stored vectors. |
| Inject | Adds the most relevant context to the prompt. |

## 🧪 Example

```text
Task:
Prepare a set of budget guidance notes for semantic retrieval.

Recommended setup:
- Mode: Embeddings
- Source: Clean text or extracted document content
- Chunking: Sentence or token-window chunks
- Storage: SQLite-backed local embeddings
- Search: Cosine similarity against the analyst question
```

## ✅ Recommended Sequence

1. Clean or normalize the source text.
2. Chunk the material into useful context units.
3. Generate embeddings for each chunk.
4. Store chunk text and vector data together.
5. Embed the user query.
6. Rank stored chunks by similarity.
7. Use top-ranked chunks as grounding context.

## ⚠️ Quality Checks

Embedding workflows should be reviewed for:

- Chunk size that is neither too small nor too large.
- Source text that preserves important headings and fiscal references.
- Stable storage of chunk text with vector data.
- Retrieval results that are relevant before they are injected into a model prompt.

## 🔗 Related API Modules

Use the `app` API documentation for local embedding utilities and the provider modules for
provider-side embedding workflows.
