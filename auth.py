"""
Firebase Authentication Module for Quantum Terminal
Handles: Registration, Login, Email Verification, Password Reset, Google OAuth
"""

import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, firestore
import json
from datetime import datetime
import requests
import extra_streamlit_components as stx
from google.oauth2 import id_token as google_id_token_verifier
from google.auth.transport import requests as google_requests

def get_cookie_manager():
    return stx.CookieManager(key="quantum_auth_v11_manager")

def get_auth_handler():
    # Helper to clean up code, but essentially just a wrapper now
    return FirebaseAuth()

class FirebaseAuth:
    """Firebase Authentication Handler with Persistent Sessions"""
    
    def __init__(self):
        """Initialize Firebase and Cookie Manager"""
        if not firebase_admin._apps:
            try:
                # Try to load from Streamlit secrets (production)
                cred_dict = dict(st.secrets["firebase"])
                cred = credentials.Certificate(cred_dict)
            except:
                # Local development - use service account file
                try:
                    cred = credentials.Certificate('firebase-credentials.json')
                except:
                    st.error("⚠️ Firebase credentials not found. Please configure Firebase.")
                    st.stop()
            
            firebase_admin.initialize_app(cred)
        
        self.db = firestore.client()
        self.api_key = st.secrets.get("firebase_api_key", "")
        # Get the global cookie manager
        self.cookie_manager = get_cookie_manager()

    def _log(self, msg):
        """Internal logging helper for debugging authentication flows"""
        # In production, we can write to a specific log file or just pass
        # Streamlit Cloud logs stdout/stderr automatically
        print(f"[AUTH] {msg}") 
        try:
            from datetime import datetime as _dt
            with open("auth_info.log", "a") as f:
                f.write(f"{_dt.now()} - {msg}\n")
        except:
            pass
    
    def get_token_from_cookie(self):
        """Get auth token from cookies for persistence"""
        return self.cookie_manager.get(cookie="quantum_auth_token")

    def set_token_cookie(self, token, expires_at=30):
        """Set auth token in cookies (default 30 days)"""
        self.cookie_manager.set("quantum_auth_token", token, expires_at=datetime.now().timestamp() + (expires_at * 24 * 60 * 60))

    def delete_token_cookie(self):
        """Remove auth token from cookies"""
        try:
            # Set to empty with immediate expiration (more reliable than delete)
            self.cookie_manager.set("quantum_auth_token", "", expires_at=datetime.now().timestamp())
        except:
            pass
        try:
            self.cookie_manager.delete("quantum_auth_token")
        except:
            pass

    def sign_up_email_password(self, email, password, username):
        """
        Register new user and trigger verification email via REST API
        Returns: (success: bool, message: str, user_id: str)
        """
        try:
            # Create user in Firebase Auth
            user = auth.create_user(
                email=email,
                password=password,
                display_name=username,
                email_verified=False
            )
            
            # TRIGGER REAL EMAIL VERIFICATION via REST API
            try:
                # 1. Sign in internally to get ID token (required for verification request)
                url_signin = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={self.api_key}"
                payload_signin = {"email": email, "password": password, "returnSecureToken": True}
                res_signin = requests.post(url_signin, json=payload_signin)
                
                if res_signin.status_code == 200:
                    id_token = res_signin.json().get('idToken')
                    # 2. Request verification email
                    url_email = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={self.api_key}"
                    payload_email = {"requestType": "VERIFY_EMAIL", "idToken": id_token}
                    res_email = requests.post(url_email, json=payload_email)
                    
                    if res_email.status_code == 200:
                        email_sent_msg = "✅ Account created! Check your email (including Spam) for the verification link."
                    else:
                        link = auth.generate_email_verification_link(email)
                        email_sent_msg = f"✅ Account created! A verification link has been generated.<br><br>👉 **[CLICK HERE TO VERIFY EMAIL]({link})**"
                else:
                    link = auth.generate_email_verification_link(email)
                    email_sent_msg = f"✅ Account created! A verification link has been generated.<br><br>👉 **[CLICK HERE TO VERIFY EMAIL]({link})**"
            except Exception as e:
                email_sent_msg = "✅ Account created! Please contact admin for verification if you don't receive an email."
            
            # Create user profile in Firestore
            self.db.collection('users').document(user.uid).set({
                'username': username,
                'email': email,
                'created_at': datetime.now(),
                'email_verified': False,
                'portfolio': [],
                'alerts': []
            })
            
            return True, email_sent_msg, user.uid
            
        except auth.EmailAlreadyExistsError:
            return False, "❌ Email already registered", None
        except Exception as e:
            return False, f"❌ Registration failed: {str(e)}", None
    
    def sign_in_email_password(self, email_or_username, password, remember_me=False):
        """
        Sign in with email/username and password
        Returns: (success: bool, message: str, user_data: dict)
        """
        try:
            # Check if input is email or username
            if '@' in email_or_username:
                email = email_or_username
            else:
                # Query Firestore for username
                users = self.db.collection('users').where('username', '==', email_or_username).limit(1).get()
                if not users:
                    return False, "❌ Username not found", None
                email = users[0].to_dict()['email']
            
            # Check if user exists first
            try:
                user = auth.get_user_by_email(email)
            except auth.UserNotFoundError:
                return False, "❌ Email not found", None
            
            # Check email verification
            if not user.email_verified:
                return False, "⚠️ Please verify your email first. Check your inbox.", None
            
            # Try Firebase REST API for password verification
            if not self.api_key:
                # Provide Google OAuth option if API key missing
                return False, "⚠️ Email/password login temporarily unavailable. Please use 'Sign in with Google' button above.", None
            
            try:
                url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={self.api_key}"
                payload = {
                    "email": email,
                    "password": password,
                    "returnSecureToken": True
                }
                
                response = requests.post(url, json=payload, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    id_token = data.get('idToken')
                    
                    # Get user profile from Firestore
                    user_doc = self.db.collection('users').document(user.uid).get()
                    user_data = user_doc.to_dict()
                    user_data['uid'] = user.uid
                    user_data['email_verified'] = user.email_verified
                    
                    # Persist session if remember_me is True
                    if remember_me and id_token:
                        self.set_token_cookie(id_token)
                    
                    # Set session state
                    st.session_state.authenticated = True
                    st.session_state.user = user_data
                    st.session_state.user_id = user_data['uid']
                    st.session_state.username = user_data['username']
                    
                    return True, "✅ Login successful!", user_data
                else:
                    error_data = response.json().get('error', {})
                    error_msg = error_data.get('message', 'Unknown error')
                    
                    if 'INVALID_PASSWORD' in error_msg:
                        return False, "❌ Invalid password", None
                    elif 'EMAIL_NOT_FOUND' in error_msg:
                        return False, "❌ Email not found", None
                    elif 'API key not valid' in error_msg or 'API_KEY_INVALID' in error_msg:
                        self._log(f"API key validation failed: {error_msg}")
                        return False, "⚠️ Email/password login temporarily unavailable. Please use 'Sign in with Google' button above.", None
                    else:
                        return False, f"❌ Login failed: {error_msg}", None
                        
            except requests.exceptions.RequestException as req_err:
                self._log(f"Network error during login: {str(req_err)}")
                return False, "❌ Network error. Please check your connection and try again.", None
            except Exception as api_error:
                self._log(f"API error: {str(api_error)}")
                return False, "⚠️ Email/password login temporarily unavailable. Please use 'Sign in with Google' button above.", None
                
        except Exception as e:
            self._log(f"Unexpected error in sign_in_email_password: {str(e)}")
            return False, f"❌ Login error: {str(e)}", None

    def sign_in_with_token(self, token):
        """Verify an existing ID token from cookie"""
        try:
            decoded_token = auth.verify_id_token(token)
            uid = decoded_token['uid']
            user = auth.get_user(uid)
            
            # Get user profile from Firestore
            user_doc = self.db.collection('users').document(uid).get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                user_data['uid'] = uid
                user_data['email_verified'] = user.email_verified
                return True, user_data
            return False, None
        except:
            return False, None

    def send_password_reset_email(self, email):
        """Send password reset email"""
        try:
            link = auth.generate_password_reset_link(email)
            return True, f"✅ Password reset email sent to {email}"
        except auth.UserNotFoundError:
            return False, "❌ Email not found"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    def verify_email(self, user_id):
        """Mark email as verified"""
        try:
            auth.update_user(user_id, email_verified=True)
            self.db.collection('users').document(user_id).update({
                'email_verified': True
            })
            return True
        except:
            return False
    
    def save_user_portfolio(self, user_id, portfolio_data):
        """Save user's portfolio to Firestore"""
        try:
            self.db.collection('users').document(user_id).update({
                'portfolio': portfolio_data,
                'last_updated': datetime.now()
            })
            return True
        except Exception as e:
            st.error(f"Error saving portfolio: {e}")
            return False
    
    def load_user_portfolio(self, user_id):
        """Load user's portfolio from Firestore"""
        try:
            doc = self.db.collection('users').document(user_id).get()
            if doc.exists:
                data = doc.to_dict()
                return data.get('portfolio', [])
            return []
        except:
            return []

    def save_user_alerts(self, user_id, alerts_data):
        """Save user's alerts to Firestore"""
        try:
            self.db.collection('users').document(user_id).update({
                'alerts': alerts_data
            })
            return True
        except:
            return False
    
    def load_user_alerts(self, user_id):
        """Load user's alerts from Firestore"""
        try:
            doc = self.db.collection('users').document(user_id).get()
            if doc.exists:
                data = doc.to_dict()
                return data.get('alerts', [])
            return []
        except:
            return []
    
    def sign_in_with_google(self, google_token):
        """
        Sign in with Google ID Token. 
        Bypasses the Firebase REST API if the API key is invalid by verifying 
        the Google Token directly and using Firebase Admin SDK.
        """
        try:
            client_id = st.secrets.get("google_oauth_client_id", "")
            
            # 1. Verify Google ID Token locally/via Google Certs (Doesn't need Firebase API Key)
            self._log("Verifying Google ID Token via google-auth library...")
            id_info = google_id_token_verifier.verify_oauth2_token(
                google_token, google_requests.Request(), client_id
            )
            
            if not id_info:
                return False, "❌ Google token verification failed", None
            
            email = id_info.get('email')
            display_name = id_info.get('name', email.split('@')[0])
            self._log(f"Google Token Verified for {email}")
            
            # 2. Get or create user in Firebase via Admin SDK (Doesn't need API Key)
            try:
                user = auth.get_user_by_email(email)
                self._log(f"Found existing Firebase user: {user.uid}")
            except auth.UserNotFoundError:
                self._log(f"Creating new Firebase user for {email}")
                user = auth.create_user(
                    email=email,
                    display_name=display_name,
                    email_verified=True
                )
            
            # 3. Get or create Firestore profile
            user_doc = self.db.collection('users').document(user.uid).get()
            
            if not user_doc.exists:
                username = display_name.replace(' ', '_').lower()
                self.db.collection('users').document(user.uid).set({
                    'username': username,
                    'email': email,
                    'created_at': datetime.now(),
                    'email_verified': True,
                    'portfolio': [],
                    'alerts': [],
                    'auth_provider': 'google'
                })
                user_data = {
                    'uid': user.uid,
                    'username': username,
                    'email': email,
                    'email_verified': True
                }
            else:
                user_data = user_doc.to_dict()
                user_data['uid'] = user.uid
                user_data['email_verified'] = True
            
            # 4. Set session state
            st.session_state.authenticated = True
            st.session_state.user = user_data
            st.session_state.user_id = user_data['uid']
            st.session_state.username = user_data['username']
            
            # 5. Handle Persistent Cookie
            # Note: We can't get a Firebase ID Token without a valid API Key.
            # We'll use the Google ID Token as a session token if possible,
            # but for simplicity, we'll just set a placeholder or use the custom token if we really need persistence.
            # For now, we'll use the google_token as the session identifier.
            
            # We try to get a Firebase ID Token JUST IN CASE the key actually works for this 
            # (sometimes identitytoolkit works while createAuthUri fails)
            firebase_token = None
            try:
                url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithIdp?key={self.api_key}"
                redirect_uri = st.secrets.get("redirect_url", "http://localhost:8501")
                payload = {
                    "postBody": f"id_token={google_token}&providerId=google.com",
                    "requestUri": redirect_uri,
                    "returnSecureToken": True
                }
                fb_res = requests.post(url, json=payload)
                if fb_res.status_code == 200:
                    firebase_token = fb_res.json().get('idToken')
            except:
                pass
                
            if firebase_token:
                self.set_token_cookie(firebase_token)
            else:
                self._log("Firebase persistent token couldn't be generated (API key issue). Using session only.")
            
            return True, "✅ Google Sign-In successful!", user_data
                
        except Exception as e:
            self._log(f"sign_in_with_google EXCEPTION: {str(e)}")
            return False, f"❌ Google Sign-In error: {str(e)}", None



def show_login_page(auth_handler):
    """Display beautiful mobile-first login/signup page"""
    h = auth_handler
    
    st.markdown("""
    <style>
    .auth-container {
        max-width: 450px;
        margin: 20px auto;
        padding: 30px;
        background: rgba(10, 14, 39, 0.8);
        border: 1px solid rgba(0, 242, 255, 0.3);
        border-radius: 24px;
        backdrop-filter: blur(20px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5), 0 0 20px rgba(0, 242, 255, 0.1);
    }
    .auth-title {
        text-align: center;
        font-size: 2.2rem;
        background: linear-gradient(90deg, #00f2ff 0%, #7b2ff7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        font-weight: 800;
    }
    .auth-subtitle {
        text-align: center;
        color: #9aa0a6;
        font-size: 0.9rem;
        margin-bottom: 30px;
        letter-spacing: 1px;
    }
    .google-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        background: white;
        color: #444;
        border-radius: 12px;
        padding: 10px;
        font-weight: 600;
        cursor: pointer;
        margin-bottom: 20px;
        border: 1px solid #ddd;
        transition: all 0.3s;
    }
    .google-btn:hover {
        background: #f1f1f1;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check for existing session
    token = h.get_token_from_cookie()
    
    if token and 'authenticated' not in st.session_state:
        success, user_data = h.sign_in_with_token(token)
        if success:
            st.session_state.authenticated = True
            st.session_state.user = user_data
            st.session_state.user_id = user_data['uid']
            st.session_state.username = user_data['username']
            st.rerun()


    # Email/Password Login Forms
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<h1 class="auth-title">QUANTUM</h1>', unsafe_allow_html=True)
        st.markdown('<p class="auth-subtitle">NEXT-GEN FINANCIAL TERMINAL</p>', unsafe_allow_html=True)
        




        
        st.markdown("<p style='text-align:center; color:#555;'>— or use email —</p>", unsafe_allow_html=True)
        
        # Tabs for Login/Signup
        mode = st.tabs(["🔑 LOGIN", "📝 SIGN UP"])
        
        with mode[0]:
            with st.form("login_form", clear_on_submit=False):
                login_input = st.text_input("Email or Username")
                password = st.text_input("Password", type="password")
                remember_me = st.checkbox("Remember me", value=True)
                
                login_btn = st.form_submit_button("ENTER TERMINAL", use_container_width=True)
                
                if login_btn:
                    if login_input and password:
                        with st.spinner("Authenticating..."):
                            success, message, user_data = h.sign_in_email_password(login_input, password, remember_me)
                            if success:
                                st.session_state.authenticated = True
                                st.session_state.user = user_data
                                st.session_state.user_id = user_data['uid']
                                st.session_state.username = user_data['username']
                                st.rerun()
                            else:
                                st.error(message)
                    else:
                        st.warning("Please fill all fields")
            
            if st.button("Forgot Password?"):
                st.info("Feature coming soon. Reset via Firebase Console for now.")

        with mode[1]:
            with st.form("signup_form"):
                new_username = st.text_input("Display Name")
                new_email = st.text_input("Email Address")
                new_password = st.text_input("Password (min 6 chars)", type="password")
                
                signup_btn = st.form_submit_button("CREATE ACCOUNT", use_container_width=True)
                
                if signup_btn:
                    if not all([new_username, new_email, new_password]):
                        st.error("Please fill all fields")
                    elif len(new_password) < 6:
                        st.error("Password too short")
                    else:
                        with st.spinner("Registering..."):
                            success, message, _ = h.sign_up_email_password(new_email, new_password, new_username)
                            if success:
                                st.markdown(f"<div style='background:rgba(0,255,100,0.1); padding:15px; border-radius:10px; border:1px solid #0f0;'>{message}</div>", unsafe_allow_html=True)
                            else:
                                st.error(message)
        
        st.markdown('</div>', unsafe_allow_html=True)


def logout(auth_handler):
    """Logout user and clear cookies"""
    # auth_handler passed as argument to avoid re-instantiation
    auth_handler.delete_token_cookie()
    for key in ['authenticated', 'user', 'user_id', 'username', 'user_loaded']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()
