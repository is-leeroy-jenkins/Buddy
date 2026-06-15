# Vector Stores

Vector stores provide reusable semantic storage for document chunks, embeddings, and provider-backed retrieval collections.

## 🧭 Purpose

Vector stores let Buddy retain searchable knowledge across requests. They support document Q&A, retrieval-augmented generation, source-grounded chat, and reusable policy or financial management corpora.

## 🧱 Workflow Position

Vector stores sit after embedding or provider file ingestion and before grounded model generation.

| Store Type | Use Case |
| --- | --- |
| Local SQLite storage | Lightweight local persistence for chunks and vectors. |
| Provider vector stores | Provider-managed file search and retrieval. |
| xAI collections | Grok collection search over uploaded or registered knowledge. |
| Gemini file search stores | Gemini-grounded retrieval workflows where supported. |

## 🧪 Example

```text
Task:
Use a stored collection of federal financial management documents to answer a policy question.

Recommended setup:
- Mode: Chat or Document Q&A
- Tool: File search, collection search, or Gemini file search
- Vector store: Select the relevant guidance or financial data store
- Prompt: Ask a specific question tied to the stored corpus
```

## ✅ Recommended Sequence

1. Identify the source corpus.
2. Ingest or register files with the appropriate storage path.
3. Confirm the vector store or collection identifier.
4. Select the retrieval tool in the provider settings.
5. Ask a specific question.
6. Review the answer and retrieved sources.
7. Update the store only when source material changes.

## ⚠️ Governance Checks

For reusable stores:

- Keep source names understandable.
- Avoid mixing unrelated corpora in one retrieval target.
- Confirm that old guidance is removed or clearly labeled.
- Do not expose API keys, file IDs, collection IDs, or sensitive content in public docs unless they are intentionally public.
- Maintain a separate inventory of source documents used to build each store.

## 🔗 Related API Modules

Use the `gpt`, `gemini`, and `grok` API documentation for provider-specific vector-store, file-search, and collection-search behavior.
