# Gemini + Google Cloud Platform (GCP) Setup Instructions

This guide explains how to enable Gemini API access, generate an API key, and create a Google Cloud
Storage bucket.



# PART 1 — Create Google Cloud Project

## 1. Create a Google Cloud Account

1. Navigate to https://console.cloud.google.com/
2. Sign in with a Google account
3. Enable billing



## 2. Create a New Project

1. Click the project selector in the top navigation
2. Click **New Project**
3. Enter a project name (e.g., buddy-gemini)
4. Click **Create**



# PART 2 — Enable Gemini API

## 3. Enable the Generative Language API

1. Navigate to https://console.cloud.google.com/apis/library
2. Search for **Generative Language API**
3. Click **Enable**



# PART 3 — Generate an API Key

## 4. Create API Key

1. Navigate to https://console.cloud.google.com/apis/credentials
2. Click **Create Credentials**
3. Select **API Key**
4. Copy the generated key

Example key format:

    AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXX



## 5. Restrict the API Key (Recommended)

1. Click the created API key
2. Under **API Restrictions**, choose **Restrict key**
3. Select **Generative Language API**
4. Save changes



# PART 4 — Create Cloud Storage Bucket

## 6. Create Bucket

1. Navigate to https://console.cloud.google.com/storage
2. Click **Create Bucket**
3. Enter a globally unique name
4. Select region
5. Keep default settings unless specific requirements exist
6. Click **Create**



## 7. Configure Permissions (If Needed)

1. Open the bucket
2. Navigate to **Permissions**
3. Add principal (user or service account)
4. Assign appropriate role (e.g., Storage Object Admin)



# PART 5 — Configure Environment Variable

### Windows (PowerShell)

    setx GOOGLE_API_KEY "AIzaSyXXXXXXXXXXXXXXXXXXXXXXXX"

Restart terminal.

### macOS / Linux

    export GOOGLE_API_KEY="AIzaSyXXXXXXXXXXXXXXXXXXXXXXXX"

To persist permanently:

    echo 'export GOOGLE_API_KEY="AIzaSyXXXXXXXXXXXXXXXXXXXXXXXX"' >> ~/.bashrc

Restart terminal.



## 8. Verify Setup

    echo $GOOGLE_API_KEY

If configured correctly, the key will display.



## Security Best Practices

- Never commit API keys to version control.
- Restrict API keys to specific APIs.
- Enable billing alerts.
- Rotate keys if exposed.