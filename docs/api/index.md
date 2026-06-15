# Document Q&A

Document Q&A supports question answering over uploaded, embedded, or provider-indexed documents.

## 🧭 Purpose

Document Q&A lets the user ground model responses in supplied documents rather than relying only on
model memory. It supports workflows where policy, guidance, financial data, legislation, or other
source material must be reviewed before an answer is produced.

## 🧱 Workflow Position

Document Q&A combines document ingestion, text extraction, chunking, embedding or provider file
search, prompt construction, and response rendering.

| Stage      | Description                                         |
|------------|-----------------------------------------------------|
| Upload     | Adds source files to the active session.            |
| Extraction | Reads text from supported document formats.         |
| Chunking   | Breaks documents into reusable context segments.    |
| Retrieval  | Finds relevant chunks or provider search results.   |
| Generation | Sends prompt plus context to the selected provider. |
| Review     | Displays answer text and available source material. |

## 📄 Supported Use Cases

| Use Case                | Example                                                             |
|-------------------------|---------------------------------------------------------------------|
| Guidance interpretation | Ask questions about OMB, Treasury, GAO, or agency policy.           |
| Budget analysis         | Query budget execution documents or financial datasets.             |
| Legislative review      | Summarize provisions from appropriations or authorization language. |
| Source comparison       | Compare policy requirements across multiple uploaded references.    |
| Briefing support        | Extract key points for analyst notes or management summaries.       |

## 🧪 Example

```text
Task:
Upload an agency budget guidance PDF and ask:
"What are the requirements for apportionment footnotes and agency reporting?"

Recommended setup:
- Mode: Document Q&A
- Provider: GPT, Gemini, or Grok
- Retrieval: File search, URL context, or local semantic context where available
- Temperature: Low
- Output: Summary with cited source references
```

## ✅ Recommended Sequence

1. Upload the source document.
2. Confirm the document appears in the active document list.
3. Select the provider and model.
4. Enable the relevant retrieval path.
5. Ask a specific question tied to the document.
6. Review source snippets and response text together.
7. Export the answer if it supports a record, briefing, or follow-up analysis.

## ⚠️ Quality Checks

Document-grounded answers should be reviewed for:

- Whether the answer references the supplied material.
- Whether unsupported assumptions were introduced.
- Whether the response distinguishes direct source content from model interpretation.
- Whether the requested source or section was actually included.
- Whether the answer is sufficiently specific for audit or policy work.

## 🔗 Related API Modules

Use the `app`, `gpt.Chat`, `gemini.Chat`, and `grok.Chat` API documentation for document handling,
retrieval options, and provider request construction.
