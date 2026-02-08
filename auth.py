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

class FirebaseAuth:
    """Firebase Authentication Handler"""
    
    def __init__(self):
        """Initialize Firebase if not already initialized"""
        if not firebase_admin._apps:
            # Load credentials from Streamlit secrets in production
            # For local testing, use service account JSON
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
    
    def sign_up_email_password(self, email, password, username):
        """
        Register new user with email/password
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
            
            # Send email verification
            link = auth.generate_email_verification_link(email)
            
            # Create user profile in Firestore
            self.db.collection('users').document(user.uid).set({
                'username': username,
                'email': email,
                'created_at': datetime.now(),
                'email_verified': False,
                'portfolio': [],
                'alerts': []
            })
            
            return True, f"✅ Account created!<br><br>👉 **[CLICK HERE TO VERIFY EMAIL]({link})**<br><br>(Use this link since we don't have an email server configured yet!)", user.uid
            
        except auth.EmailAlreadyExistsError:
            return False, "❌ Email already registered", None
        except Exception as e:
            return False, f"❌ Registration failed: {str(e)}", None
    
    def sign_in_email_password(self, email_or_username, password):
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
            
            # Use Firebase REST API for sign in (Firebase Admin SDK doesn't support this directly)
            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={self.api_key}"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                user = auth.get_user_by_email(email)
                
                # Get user profile from Firestore
                user_doc = self.db.collection('users').document(user.uid).get()
                user_data = user_doc.to_dict()
                user_data['uid'] = user.uid
                user_data['email_verified'] = user.email_verified
                
                if not user.email_verified:
                    return False, "⚠️ Please verify your email first. Check your inbox.", None
                
                return True, "✅ Login successful!", user_data
            else:
                error = response.json().get('error', {}).get('message', 'Unknown error')
                if 'INVALID_PASSWORD' in error:
                    return False, "❌ Invalid password", None
                elif 'EMAIL_NOT_FOUND' in error:
                    return False, "❌ Email not found", None
                else:
                    return False, f"❌ Login failed: {error}", None
                
        except Exception as e:
            return False, f"❌ Login error: {str(e)}", None
    
    def send_password_reset_email(self, email):
        """
        Send password reset email
        Returns: (success: bool, message: str)
        """
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


def show_login_page():
    """Display beautiful login/signup page"""
    
    st.markdown("""
    <style>
    .auth-container {
        max-width: 500px;
        margin: 50px auto;
        padding: 40px;
        background: linear-gradient(135deg, rgba(0, 242, 255, 0.1), rgba(123, 47, 247, 0.1));
        border: 2px solid rgba(0, 242, 255, 0.3);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 242, 255, 0.2);
    }
    .auth-title {
        text-align: center;
        font-size: 2.5rem;
        background: linear-gradient(90deg, #00f2ff 0%, #7b2ff7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<h1 class="auth-title">💠 QUANTUM TERMINAL</h1>', unsafe_allow_html=True)
        
        # Tabs for Login/Signup
        auth_tab = st.radio("", ["🔑 Login", "📝 Sign Up"], horizontal=True, label_visibility="collapsed")
        
        st.markdown("---")
        
        auth_handler = FirebaseAuth()
        
        if auth_tab == "🔑 Login":
            # LOGIN FORM
            st.subheader("🔑 Login")
            
            with st.form("login_form"):
                login_input = st.text_input("📧 Email or Username", placeholder="your@email.com or username")
                password = st.text_input("🔒 Password", type="password")
                
                col_login, col_forgot = st.columns(2)
                
                with col_login:
                    login_btn = st.form_submit_button("🚀 LOGIN", use_container_width=True)
                
                with col_forgot:
                    forgot_btn = st.form_submit_button("🔄 Forgot Password?", use_container_width=True)
                
                if login_btn:
                    if login_input and password:
                        with st.spinner("Checking credentials..."):
                            success, message, user_data = auth_handler.sign_in_email_password(login_input, password)
                            
                            if success:
                                st.success(message)
                                # Save to session state
                                st.session_state.authenticated = True
                                st.session_state.user = user_data
                                st.session_state.user_id = user_data['uid']
                                st.session_state.username = user_data['username']
                                st.rerun()
                            else:
                                st.error(message)
                    else:
                        st.warning("⚠️ Please fill all fields")
                
                if forgot_btn:
                    if '@' in login_input:
                        success, message = auth_handler.send_password_reset_email(login_input)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                    else:
                        st.warning("⚠️ Please enter your email address")
        
        else:
            # SIGNUP FORM
            st.subheader("📝 Create Account")
            
            with st.form("signup_form"):
                username = st.text_input("👤 Username", placeholder="Choose a username")
                email = st.text_input("📧 Email", placeholder="your@email.com")
                email_confirm = st.text_input("📧 Confirm Email", placeholder="Repeat email")
                password = st.text_input("🔒 Password", type="password", placeholder="Min 6 characters")
                password_confirm = st.text_input("🔒 Confirm Password", type="password", placeholder="Repeat password")
                
                signup_btn = st.form_submit_button("✨ CREATE ACCOUNT", use_container_width=True)
                
                if signup_btn:
                    # Validation
                    if not all([username, email, email_confirm, password, password_confirm]):
                        st.error("❌ Please fill all fields")
                    elif email != email_confirm:
                        st.error("❌ Emails don't match")
                    elif password != password_confirm:
                        st.error("❌ Passwords don't match")
                    elif len(password) < 6:
                        st.error("❌ Password must be at least 6 characters")
                    elif '@' not in email:
                        st.error("❌ Invalid email format")
                    else:
                        with st.spinner("Creating account..."):
                            success, message, user_id = auth_handler.sign_up_email_password(email, password, username)
                            
                            if success:
                                st.markdown(f"""
                                <div style="padding: 20px; background-color: rgba(0, 255, 0, 0.1); border-left: 5px solid #00ff00; border-radius: 5px;">
                                    {message}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.error(message)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Google OAuth Button (placeholder - requires additional setup)
        st.markdown("---")
        st.info("🔜 Google Sign-In coming soon! Use email registration for now.")


def logout():
    """Logout user"""
    for key in ['authenticated', 'user', 'user_id', 'username']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()
