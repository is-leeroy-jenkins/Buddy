# Utilities

Utilities mode collects supporting runtime controls used to configure, inspect, reset, and maintain the application session.

## 🧭 Purpose

Utility workflows help keep the application stable during iterative analysis. They support configuration review, runtime inspection, state reset, environment verification, and operational cleanup without changing the primary model or data workflows.

## 🧱 Workflow Position

Utilities sit beside the main modes. They are used before setup, during troubleshooting, or after a session when state cleanup is needed.

| Utility Area | Use Case |
| --- | --- |
| Runtime configuration | Confirm provider keys, model defaults, or environment settings. |
| Session inspection | Review active state, selected provider, mode, or model values. |
| Reset controls | Clear chat history, active documents, generated outputs, or temporary settings. |
| Environment checks | Confirm required dependencies, directories, or API key availability. |
| Diagnostics | Support troubleshooting without editing source code. |

## 🧪 Example

```text
Task:
Reset the session before starting a new analysis.

Recommended sequence:
- Export needed results first
- Clear active chat or document state
- Confirm provider and mode settings
- Start the next workflow with clean context
```

## ✅ Recommended Sequence

1. Export or save anything needed from the current session.
2. Use reset controls only for the state you intend to clear.
3. Confirm provider keys and selected mode.
4. Reopen documents, prompts, or stores as needed.
5. Run a small test request before a larger analysis.

## ⚠️ Operational Checks

Use utility actions carefully:

- Do not clear session state before exporting important results.
- Confirm whether reset actions affect chat, documents, prompts, or generated artifacts.
- Avoid changing environment variables while active provider sessions are running.
- Restart the Streamlit app when dependency or environment changes require a clean runtime.

## 🔗 Related API Modules

Use the `app` API documentation for session-state initialization, reset helpers, runtime configuration, and UI utility functions.
