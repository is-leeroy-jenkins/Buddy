# xAI (Grok) API Setup Instructions

This guide explains how to create an xAI account, generate a Grok API key, and configure it for
local development.



## 1. Create an xAI Account

1. Navigate to https://x.ai/
2. Click **Sign Up**
3. Register using X (Twitter) or email
4. Complete account verification



## 2. Enable API Access

1. Navigate to the xAI Developer Portal
2. Locate **API Access** or **Developer Settings**
3. Enable API access and accept terms



## 3. Configure Billing (If Required)

1. Navigate to Billing Settings
2. Add a payment method
3. Review pricing and usage limits



## 4. Generate an API Key

1. Navigate to the API Keys section
2. Click **Create API Key**
3. Name the key (e.g., buddy-grok)
4. Copy the generated key

Example key format:

    xai-xxxxxxxxxxxxxxxxxxxxxxxx

Store securely.



## 5. Configure Environment Variable

### Windows (PowerShell)

    setx GROK_API_KEY "xai-xxxxxxxxxxxxxxxxxxxxxxxx"

Restart terminal.

### macOS / Linux

    export GROK_API_KEY="xai-xxxxxxxxxxxxxxxxxxxxxxxx"

To persist permanently:

    echo 'export GROK_API_KEY="xai-xxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc

Restart terminal.



## 6. Verify Setup

    echo $GROK_API_KEY

If configured correctly, the key will display.



## Security Best Practices

- Never commit API keys to version control.
- Do not embed keys in source files.
- Rotate keys if exposed.