# OpenAI API Setup Instructions

This guide explains how to create an OpenAI account, generate an API key, and configure it for local
development.



## 1. Create an OpenAI Account

1. Navigate to https://platform.openai.com/
2. Click **Sign Up**
3. Register using email/password or SSO
4. Verify your email
5. Complete onboarding



## 2. Configure Billing

1. Go to https://platform.openai.com/account/billing
2. Click **Add payment method**
3. Enter payment details
4. (Recommended) Set usage limits to control costs



## 3. Generate an API Key

1. Navigate to https://platform.openai.com/api-keys
2. Click **Create new secret key**
3. Name the key (e.g., buddy-app)
4. Copy the key immediately

Example key format:

    sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Store the key securely. It will not be displayed again.



## 4. Configure Environment Variable

### Windows (PowerShell)

    setx OPENAI_API_KEY "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

Restart the terminal.

### macOS / Linux

    export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

To persist permanently:

    echo 'export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc

Restart the terminal.



## 5. Verify Setup

    echo $OPENAI_API_KEY

If configured correctly, the key will display.



## Security Best Practices

- Never commit API keys to version control.
- Do not hardcode keys in source files.
- Use environment variables or `.env` files.
- Rotate keys immediately if exposed.