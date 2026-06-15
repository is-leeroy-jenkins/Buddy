# User Guide

Buddy is a Streamlit-based federal financial management assistant for structured large-language-model workflows, document-grounded analysis, semantic search, prompt governance, and local data inspection.

## 🧭 Purpose

This guide explains how to operate Buddy from the perspective of an analyst using the application. It focuses on the major runtime modes exposed by the interface: chat, document Q&A, image generation, audio processing, embeddings, vector stores, data management, prompt management, and export/reporting.

## 🧱 Workflow Position

Buddy sits between the analyst and multiple model/provider services. The user selects a provider, configures model behavior, supplies prompts or files, and reviews generated text, sources, artifacts, tables, or exports.

| Area | Use When |
| --- | --- |
| Chat | You need structured model responses with configurable provider settings. |
| Document Q&A | You need answers grounded in uploaded or embedded documents. |
| Image Generation | You need provider-backed image creation from a prompt. |
| Audio | You need speech, transcription, or translation workflows. |
| Embeddings | You need vector representations for semantic retrieval. |
| Vector Stores | You need reusable retrieval stores or provider-side collections. |
| Data Management | You need to inspect, profile, filter, visualize, or modify local SQLite data. |
| Prompt Management | You need governed system instructions or reusable prompt templates. |
| Export & Reporting | You need Markdown, XML, PDF, or downloadable analysis outputs. |

## ✅ Recommended Sequence

1. Configure provider API keys before starting model workflows.
2. Select the provider and mode that matches the task.
3. Confirm model, token, reasoning, tool, and storage settings.
4. Add source documents, context, URLs, domains, or vector-store IDs where needed.
5. Submit the prompt or action.
6. Review the response, sources, generated artifacts, and token usage.
7. Export results when the output needs to be reused, archived, or shared.

## 🧪 Example Session

```text
1. Start Buddy with streamlit run app.py.
2. Select GPT, Gemini, or Grok as the provider.
3. Choose Chat or Document Q&A.
4. Enter a budget, fiscal law, or financial management question.
5. Enable file search, web search, URL context, or collection search when grounding is needed.
6. Review generated text and source references before using the result.
```

## 🔗 Related API Modules

The user guide is supported by the source-level API documentation for `app`, `gpt`, `gemini`, and `grok`.
