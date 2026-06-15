# Chat

Chat mode provides the primary analyst-facing interaction path for provider-backed text generation.

## 🧭 Purpose

Chat mode lets the user submit structured prompts to GPT, Gemini, or Grok while controlling
provider, model, system instructions, reasoning, token limits, response format, tool usage, and
conversational context. It is the recommended entry point for budget guidance questions, policy
interpretation, source-grounded summaries, and analytical drafting.

## 🧱 Workflow Position

Chat mode is the top-level interaction layer. It collects user input, model settings, tool options,
and session context, then routes the request through the selected provider wrapper.

| Step | Runtime Action |
| --- | --- |
| Provider selection | Chooses GPT, Gemini, or Grok. |
| Model selection | Applies the provider-specific model option. |
| Prompt entry | Captures the analyst question or task. |
| Tool configuration | Adds web search, file search, URL context, code execution, or collection search where supported. |
| Response rendering | Displays generated text, source references, artifacts, and usage metadata. |

## 🛠 Configuration Areas

| Setting | Guidance |
| --- | --- |
| Model | Use the smallest model that satisfies the accuracy and reasoning requirement. |
| System instructions | Use governed prompts for repeatable behavior. |
| Temperature | Keep low for policy, financial, and audit-support answers. |
| Max tokens | Set high enough for complete answers but avoid unnecessary cost. |
| Reasoning | Use higher settings for multi-step analysis when supported by the provider. |
| Tools | Enable only the tools needed for the current question. |
| Store/background/stream | Use only when supported and needed for the workflow. |

## 🧪 Example

```text
Question:
Explain how the apportionment process controls the rate of obligation for budgetary resources.

Recommended setup:
- Provider: GPT or Gemini
- Mode: Chat
- Temperature: Low
- System instructions: Budget guidance analyst
- Tools: File search or web search if current source grounding is required
```

## ✅ Recommended Sequence

1. Select the provider.
2. Select the chat-capable model.
3. Load or enter system instructions.
4. Configure tools only when grounding is required.
5. Enter the analyst prompt.
6. Submit the request.
7. Review the answer, citations, generated artifacts, and token usage.
8. Save or export the result if it supports a briefing, memo, or audit trail.

## ⚠️ Review Checks

Before relying on a generated answer, verify:

- The model used the intended provider and model.
- Source references are present when the answer depends on external documents.
- The response distinguishes guidance from calculation.
- The output does not rely on stale assumptions.
- Any generated table or artifact matches the question asked.

## 🔗 Related API Modules

Use the `app`, `gpt.Chat`, `gemini.Chat`, and `grok.Chat` API documentation for implementation
details.
