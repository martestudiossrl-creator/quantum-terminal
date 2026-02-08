# 🚀 QUANTUM TERMINAL - QUICK DEPLOYMENT GUIDE

## ⚡ FASE 1: SETUP FIREBASE (5 minuti)

### Passo 1: Crea Progetto Firebase
1. Vai su: https://console.firebase.google.com/
2. Click **"Aggiungi progetto"**
3. Nome: `quantum-terminal`
4. Disabilita Google Analytics
5. Click **"Crea progetto"**

### Passo 2: Attiva Autenticazione  
1. Menu laterale → **Authentication** → **Inizia**
2. Tab **"Metodo di accesso"**
3. Abilita **"Email/Password"** → Salva

### Passo 3: Crea Database Firestore
1. Menu laterale → **Firestore Database** → **"Crea database"**
2. Seleziona **"Inizia in modalità produzione"**
3. Località: **europe-west**
4. Click **"Attiva"**

### Passo 4: Configura Regole Firestore
1. Tab **"Regole"**
2. Incolla questo codice:
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
3. **Pubblica**

### Passo 5: Scarica Credenziali
1. ⚙️ **Impostazioni progetto** (icona ingranaggio in alto)
2. Tab **"Account di servizio"**
3. Click **"Genera nuova chiave privata"**
4. Scarica il file JSON
5. **Rinominalo** in `firebase-credentials.json`
6. **Spostalo** nella cartella `quantum_terminal_v11`

### Passo 6: Ottieni API Key
1. ⚙️ **Impostazioni progetto** → Tab **"Generali"**
2. Scroll fino a "Le tue app"
3. Click icona **Web** `</>`
4. Nome app: `quantum-terminal-web`
5. **COPIA** la chiave API (inizia con `AIza...`)

### Passo 7: Crea File Secrets
1. Nella cartella `quantum_terminal_v11`, crea cartella `.streamlit`
2. Dentro `.streamlit`, crea file `secrets.toml`
3. Apri `secrets.toml` e incolla:

```toml
firebase_api_key = "INCOLLA_QUI_LA_API_KEY"

[firebase]
type = "service_account"
project_id = "COPIA_DA_JSON"
private_key_id = "COPIA_DA_JSON"
private_key = "COPIA_DA_JSON_CON_BACKSLASH_N"
client_email = "COPIA_DA_JSON"
client_id = "COPIA_DA_JSON"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "COPIA_DA_JSON"
```

4. **Sostituisci** tutti i valori copiandoli da `firebase-credentials.json`

---

## ⚡ FASE 2: TEST LOCALE (2 minuti)

### Installa Dipendenze
```bash
cd C:\Users\gbran\.gemini\antigravity\scratch\quantum_terminal_v11
pip install -r requirements.txt
```

### Avvia App
```bash
python -m streamlit run app.py
```

### Testa Autenticazione
1. Apri http://localhost:8501
2. Click **"Sign Up"**
3. Crea account di test
4. Controlla email per verifica
5. Click link di verifica
6. Torna all'app e fai login

---

## ⚡ FASE 3: DEPLOY ONLINE (3 minuti)

### Opzione A: Streamlit Cloud (GRATUITO)

1. **Crea Repository GitHub:**
   - Vai su https://github.com/new
   - Nome: `quantum-terminal`
   - Public o Private
   - Create repository

2. **Push Codice:**
   ```bash
   cd C:\Users\gbran\.gemini\antigravity\scratch\quantum_terminal_v11
   git init
   git add .
   git commit -m "Initial commit - Quantum Terminal"
   git remote add origin https://github.com/TUO_USERNAME/quantum-terminal.git
   git push -u origin main
   ```

3. **Deploy su Streamlit Cloud:**
   - Vai su https://share.streamlit.io/
   - Click **"New app"**
   - Connetti GitHub
   - Seleziona repository `quantum-terminal`
   - Main file: `app.py`
   - Click **"Advanced settings"**
   - Incolla tutto il contenuto di `.streamlit/secrets.toml` nella sezione Secrets
   - Click **"Deploy!"**

4. **FATTO!** Link pubblico disponibile in 2-3 minuti

---

## 🎯 SUMMARY - Cosa Hai Ottenuto

✅ **Autenticazione Completa:**
- Registrazione con email/password
- Verifica email automatica
- Login con email o username
- Password reset via email
- Database utenti sicuro

✅ **Area Personale:**
- Portfolio salvato per ogni utente
- Alerts personalizzati
- Dati persistenti

✅ **Web App Pubblica:**
- Accessibile da qualsiasi dispositivo
- HTTPS sicuro
- 100% gratuito (Streamlit Cloud free tier)
- Uptime 99.9%

---

## 🆘 TROUBLESHOOTING

**Errore: "Firebase credentials not found"**
→ Verifica che `firebase-credentials.json` sia nella cartella giusta

**Email di verifica non arriva**
→ Controlla spam, attendi 1-2 minuti

**Errore al deploy su Streamlit Cloud** 
→ Verifica che i secrets siano copiati correttamente

**App lenta al primo caricamento**
→ Normale! Streamlit Cloud free tier ha cold start

---

## 📞 NEXT STEPS

1. Personalizza email templates in Firebase Console
2. Aggiungi Google OAuth (opzionale)
3. Configura dominio custom (opzionale, richiede piano a pagamento)
4. Monitora utilizzo in Firebase Console

---

**PRONTO PER INIZIARE?**  
Segui FASE 1 → FASE 2 → FASE 3  
Tempo totale: ~10 minuti 🚀
