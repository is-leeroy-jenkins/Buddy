# Prompt Management

Prompt Management supports controlled creation, editing, versioning, loading, and export of reusable system instructions and prompt templates.

## 🧭 Purpose

Prompt Management lets analysts maintain repeatable instructions for provider workflows. It stores prompt metadata and text in SQLite so that system instructions can be reused, revised, exported, and applied consistently across model sessions.

## 🧱 Workflow Position

Prompt Management sits before model execution. It prepares the instruction layer that guides Chat, Document Q&A, and other provider-backed tasks.

| Field | Purpose |
| --- | --- |
| Caption | Display name used in the application selector. |
| Name | Internal or descriptive prompt name. |
| Text | System instruction or prompt body. |
| Version | Version label for governance and controlled revision. |
| ID | External or internal identifier for traceability. |

## 🧪 Example

```text
Task:
Create a reusable federal budget analyst system instruction.

Recommended setup:
- Caption: Budget Analyst
- Name: Federal Budget Guidance Analyst
- Version: 1.0
- Text: Define role, source hierarchy, tone, citation expectations, and analysis limits
- ID: Stable local prompt identifier
```

## ✅ Recommended Sequence

1. Open Prompt Management.
2. Select an existing prompt or create a new one.
3. Enter caption, name, text, version, and ID.
4. Save the prompt.
5. Load the prompt into the active system-instruction state.
6. Use the prompt in Chat or Document Q&A.
7. Export prompt text when documentation, review, or portability is needed.

## 🔄 XML and Markdown Conversion

Buddy supports converting structured instruction text between XML-delimited sections and Markdown headings. Use XML-style structure when strict prompt segmentation is needed. Use Markdown when the instructions are intended for documentation, review, or readable export.

```xml
<role>
Federal budget analyst.
</role>

<rules>
Use source-grounded answers where available.
</rules>
```

```markdown
## Role

Federal budget analyst.

## Rules

Use source-grounded answers where available.
```

## ⚠️ Governance Checks

Prompt templates should be reviewed for:

- Clear role definition.
- Source hierarchy and citation expectations.
- Limits on unsupported assumptions.
- Version identifiers.
- No embedded secrets, API keys, or sensitive source text.
- Compatibility with all intended providers.

## 🔗 Related API Modules

Use the `app` API documentation for prompt storage, prompt retrieval, prompt updates, deletion, and XML/Markdown conversion utilities.
