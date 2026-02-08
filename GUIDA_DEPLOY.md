# 🚀 QUANTUM TERMINAL - QUICK DEPLOYMENT GUIDE (PRO READY)

## ⚡ FASE 1: SETUP FIREBASE (5 minuti)

### Passo 1: Crea Progetto Firebase
1. Vai su: [Firebase Console](https://console.firebase.google.com/)
2. Click **"Aggiungi progetto"**. Nome: `quantum-terminal`.

### Passo 2: Attiva Autenticazione  
1. Menu laterale → **Authentication** → **Inizia**.
2. Tab **"Metodo di accesso"** → Abilita **"Email/Password"** e **"Google"**.

### Passo 3: Crea Database Firestore
1. Menu laterale → **Firestore Database** → **"Crea database"**. Seleziona **europe-west**.

### Passo 4: Scarica Credenziali (CRUCIALE)
1. ⚙️ **Impostazioni progetto** → Tab **"Account di servizio"**.
2. Click **"Genera nuova chiave privata"**. 
3. Scarica il file JSON e rinominalo in `firebase-credentials.json`.
4. Copia i valori nel file `.streamlit/secrets.toml`.

### Passo 5: Google OAuth
1. In Firebase Console → Authentication → Google, copia il **Web Client ID**.
2. Aggiungi il Client ID e il Client Secret (trovabile su [Google Cloud Console](https://console.cloud.google.com/apis/credentials)) nei segreti.

---

## ⚡ FASE 2: DEPLOY ONLINE

### Opzione: Streamlit Cloud (GRATUITO)

1. **Carica su GitHub**: Carica tutto tranne `firebase-credentials.json` e `secrets.toml` (sono già nel `.gitignore`).
2. **Deploy**: Su Streamlit Cloud, incolla il contenuto di `secrets.toml` nella sezione **Advanced Settings > Secrets**.
3. **Redirect URI**: Ricordati di aggiungere `https://tua-app.streamlit.app` negli "Authorized redirect URIs" della Google Cloud Console.

---

## 🎯 PERCHÉ QUESTA VERSIONE È SUPERIORE
✅ **Robust Auth**: Ora gestisce automaticamente gli errori di duplicazione di Streamlit.
✅ **Bypass Key**: Il login con Google funziona anche se la chiave API di Firebase ha problemi, grazie alla verifica diretta.
✅ **Auto-Clean**: Pulisce l'URL dopo il login per una navigazione pulita.
✅ **Persistence**: Resti loggato anche se chiudi il browser.

L'app è pronta. 🚀
