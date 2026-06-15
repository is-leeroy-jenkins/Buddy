# Image Generation

Image generation mode supports prompt-driven visual output through configured providers.

## 🧭 Purpose

Image generation lets the user create visual assets from text prompts while controlling provider, model, style, output size, quality, and related generation settings. It is useful for documentation visuals, architecture diagrams, concept illustrations, and presentation-support graphics.

## 🧱 Workflow Position

Image generation sits after prompt planning and before documentation or presentation assembly. The user defines the visual intent, configures provider options, generates the image, and reviews the output before saving or embedding it.

| Stage | Description |
| --- | --- |
| Prompt design | Describes the visual objective, content, style, and constraints. |
| Provider setup | Selects image-capable provider and model. |
| Generation settings | Controls size, quality, background, style, and count where supported. |
| Output review | Checks whether the image matches the documentation or presentation need. |
| Reuse | Saves the image into documentation, README, or project assets. |

## 🧪 Example

```text
Prompt:
Create a dark-mode architecture diagram for Buddy showing Streamlit UI, provider wrappers,
SQLite storage, document processing, embeddings, vector stores, logging, and exports.

Recommended setup:
- Mode: Image Generation
- Background: Dark
- Aspect ratio: 16:9 when available
- Style: Technical architecture diagram
- Output target: docs/images/buddy-architecture-dark.png
```

## ✅ Recommended Sequence

1. Write the image prompt with the exact title, modules, and style.
2. Select the image-capable provider and model.
3. Set size, quality, and background options.
4. Generate the image.
5. Review for missing modules or misleading relationships.
6. Save with a stable documentation filename.
7. Reference the image from Markdown only after the file exists in the docs folder.

## ⚠️ Documentation Checks

Before adding generated images to MkDocs:

- Confirm the image file exists under `docs/images/`.
- Use forward slashes in Markdown paths.
- Avoid referencing images that are still only in `resources/` unless that path is included in the docs build.
- Use dark-mode diagrams when the site theme is dark.
- Keep diagrams structural and readable; avoid decorative clutter.

## 🔗 Related API Modules

Use the `app`, `gpt`, `gemini`, and `grok` API documentation for provider options that support image workflows.
