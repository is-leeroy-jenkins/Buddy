## Set `OPENAI_API_KEY` Environment Variable

Follow the steps below to set the `OPENAI_API_KEY` environment variable so your application can
authenticate to OpenAI.

---

### Windows (PowerShell) — Permanent (Recommended)

1. Open **PowerShell** (not Command Prompt).

2. Run the command below, replacing the placeholder with your real key:

   setx OPENAI_API_KEY "sk-REPLACE_WITH_YOUR_KEY"

3. **Close** PowerShell completely.

4. Re-open PowerShell.

5. Verify the variable is set:

   echo $env:OPENAI_API_KEY

If configured correctly, PowerShell will print your key.

---

### Windows (PowerShell) — Current Session Only

Use this if you only want the key available until you close the terminal.

1. Open **PowerShell**.

2. Set the variable for the current session:

   $env:OPENAI_API_KEY = "sk-REPLACE_WITH_YOUR_KEY"

3. Verify:

   echo $env:OPENAI_API_KEY

---

### Windows (Command Prompt) — Current Session Only

1. Open **Command Prompt**.

2. Set the variable:

   set OPENAI_API_KEY=sk-REPLACE_WITH_YOUR_KEY

3. Verify:

   echo %OPENAI_API_KEY%

---

### macOS / Linux (Bash / Zsh) — Current Session Only

1. Open a terminal.

2. Set the variable:

   export OPENAI_API_KEY="sk-REPLACE_WITH_YOUR_KEY"

3. Verify:

   echo $OPENAI_API_KEY

---

### macOS / Linux (Bash / Zsh) — Permanent (Recommended)

1. Open a terminal.

2. Add the export line to your shell profile:

   echo 'export OPENAI_API_KEY="sk-REPLACE_WITH_YOUR_KEY"' >> ~/.bashrc

   If you use Zsh (common on macOS), use:

   echo 'export OPENAI_API_KEY="sk-REPLACE_WITH_YOUR_KEY"' >> ~/.zshrc

3. Reload your shell profile:

   source ~/.bashrc

   or:

   source ~/.zshrc

4. Verify:

   echo $OPENAI_API_KEY

---

### Security Notes

* Do not commit keys to source control.
* Do not store keys directly in code.
* Rotate your key immediately if exposed.
