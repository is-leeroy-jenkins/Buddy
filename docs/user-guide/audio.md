# Audio Processing

Audio processing mode supports speech-related workflows such as text-to-speech, transcription, and
translation where provider support is available.

## 🧭 Purpose

Audio processing lets the user convert text to speech, transcribe uploaded audio, or translate
spoken content while retaining provider-specific configuration in the application interface. It
supports review, accessibility, briefing preparation, and multimodal analysis workflows.

## 🧱 Workflow Position

Audio mode is a provider-backed multimodal workflow. It collects model, voice, language, file,
timing, and playback settings, then routes the request through the selected provider capability.

| Workflow | Description |
| --- | --- |
| Text-to-Speech | Converts text into spoken audio. |
| Transcription | Converts speech audio into text. |
| Translation | Converts speech from one language into translated text where supported. |
| Playback | Supports user review of generated or uploaded audio. |
| Export | Saves generated output when the provider and UI path support it. |

## 🧪 Example

```text
Task:
Convert a short budget briefing paragraph into spoken audio for review.

Recommended setup:
- Mode: Audio
- Task: Text-to-Speech
- Voice: Select a supported provider voice
- Input: Final briefing paragraph
- Review: Play the output before reuse
```

## ✅ Recommended Sequence

1. Select the provider and audio-capable model.
2. Select the audio task.
3. Provide text or upload an audio file.
4. Set voice, language, timing, or playback options.
5. Run the audio operation.
6. Review generated text or audio.
7. Export only after confirming the output is accurate.

## ⚠️ Review Checks

For transcription or translation:

- Verify proper nouns, acronyms, and agency names.
- Review numbers, fiscal years, and dollar amounts.
- Confirm that punctuation and paragraph breaks are usable.
- Avoid relying on raw transcripts for final policy language without review.

## 🔗 Related API Modules

Use the `app`, `gpt`, `gemini`, and `grok` API documentation for provider-specific audio options.
