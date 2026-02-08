# Firebase Configuration Guide

## Setup Instructions

### 1. Create Firebase Project
1. Go to https://console.firebase.google.com/
2. Click "Add project"
3. Enter project name: "quantum-terminal"
4. Disable Google Analytics (optional)
5. Click "Create project"

### 2. Enable Authentication Methods
1. In Firebase Console, go to **Authentication** > **Get Started**
2. Go to **Sign-in method** tab
3. Enable:
   - ✅ **Email/Password**
   - ✅ **Google** (optional, for OAuth)

### 3. Create Firestore Database
1. Go to **Firestore Database** > **Create database**
2. Select **Start in production mode**
3. Choose location (europe-west or us-central)
4. Click **Enable**

### 4. Security Rules (Firestore)
Go to **Firestore** > **Rules** and paste:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
  }
}
```

### 5. Get Service Account Credentials
1. Go to **Project Settings** (⚙️ icon) > **Service accounts**
2. Click **Generate new private key**
3. Download the JSON file
4. Save as `firebase-credentials.json` in project root

### 6. Get Web API Key
1. Go to **Project Settings** > **General**
2. Scroll to "Your apps" section
3. Click "Web app" (</> icon)
4. Register app name: "quantum-terminal-web"
5. Copy the **API Key** (starts with `AIza...`)

### 7. Configure Streamlit Secrets

Create `.streamlit/secrets.toml` with:

```toml
firebase_api_key = "YOUR_WEB_API_KEY_HERE"

[firebase]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----\n"
client_email = "firebase-adminsdk-xxxxx@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-xxxxx%40your-project.iam.gserviceaccount.com"
```

Replace all values with data from `firebase-credentials.json`

### 8. Email Templates (Optional)
1. Go to **Authentication** > **Templates**
2. Customize:
   - Email verification
   - Password reset
   - Email change

### 9. Deploy to Streamlit Cloud
1. Push code to GitHub repository
2. Go to https://share.streamlit.io/
3. Connect GitHub account
4. Select repository
5. Add secrets from step 7 to Streamlit Cloud secrets
6. Deploy!

## Environment Variables Needed

### Local Development:
- `firebase-credentials.json` file in root

### Production (Streamlit Cloud):
Add these to **App settings** > **Secrets**:
- All content from `.streamlit/secrets.toml`

## Testing

Test the authentication:
```bash
python -m streamlit run app.py
```

1. Create account with email/password
2. Check email for verification link
3. Click verification link
4. Login with verified email
5. Access should work!

## Troubleshooting

**Error: "Firebase credentials not found"**
- Make sure `firebase-credentials.json` exists
- Or secrets are properly configured in Streamlit Cloud

**Error: "Email already exists"**
- User already registered, use password reset
- Or check Firebase Console > Authentication > Users

**Verification email not received**
- Check spam folder
- Verify sender email in Firebase > Authentication > Templates
- Wait 1-2 minutes

**Can't login after verification**
- Make sure you clicked the verification link in email
- Check Firebase Console if email_verified = true

## Security Notes

⚠️ **IMPORTANT**: 
- Never commit `firebase-credentials.json` to Git
- Add to `.gitignore`
- Use Streamlit secrets in production
- Enable Firestore security rules
- Use HTTPS only in production
