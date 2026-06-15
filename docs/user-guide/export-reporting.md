# Export & Reporting

Export and reporting workflows support reuse of generated instructions, chat history, markdown notes, and PDF-ready outputs.

## 🧭 Purpose

Export workflows let analysts move Buddy outputs into documentation, review packets, briefing support, archival notes, or external reporting formats. The export path supports controlled reuse of system prompts, chat history, and model-generated analytical material.

## 🧱 Workflow Position

Export sits at the end of an analysis workflow. After the user has reviewed model output, source references, and artifacts, the result can be saved in a portable format.

| Export Type | Use Case |
| --- | --- |
| Markdown | Documentation, notes, GitHub, MkDocs, and review drafts. |
| XML-delimited instructions | Structured prompt reuse and system-instruction portability. |
| PDF | Briefing, archival, or distribution-ready output. |
| Downloaded artifacts | Tables, text files, or generated data outputs from model workflows. |

## 🧪 Example

```text
Task:
Export a reviewed chat session for documentation.

Recommended setup:
- Review final response
- Confirm source references
- Export chat history as Markdown
- Save generated artifacts separately
- Add the reviewed result to project documentation or analysis records
```

## ✅ Recommended Sequence

1. Complete the model or data workflow.
2. Review the generated text and sources.
3. Confirm the output is final enough to preserve.
4. Choose Markdown, XML, PDF, or artifact download.
5. Save the file with a stable name.
6. Store exports with the related project, prompt, or analysis package.

## ⚠️ Review Checks

Before exporting:

- Remove test prompts or abandoned analysis turns.
- Confirm no API keys, tokens, or sensitive identifiers are included.
- Verify tables and calculations.
- Confirm generated prose aligns with the final source material.
- Keep exported prompts versioned when used operationally.

## 🔗 Related API Modules

Use the `app` API documentation for export utilities, prompt conversion, chat history, PDF generation, and downloadable artifacts.
