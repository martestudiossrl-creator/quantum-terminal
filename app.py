import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import feedparser
from datetime import datetime, timedelta

import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy
# Import custom modules
from asset_database import search_assets, get_all_tickers, get_categories, get_assets_by_category
from ai_assistant import get_ai_response, get_chart_tips, get_fundamental_tips
import auth

# PAGE CONFIG
st.set_page_config(page_title="QUANTUM TERMINAL V11 PRO", layout="wide", page_icon="💠")

# AUTHENTICATION CHECK
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    auth.show_login_page()
    st.stop()

# LOAD USER DATA IF JUST LOGGED IN
if 'portfolio' not in st.session_state or (st.session_state.get('user_loaded') != st.session_state.user_id):
    auth_handler = auth.FirebaseAuth()
    
    # Load Portfolio
    user_portfolio = auth_handler.load_user_portfolio(st.session_state.user_id)
    if user_portfolio:
        st.session_state.portfolio = pd.DataFrame(user_portfolio)
    else:
        st.session_state.portfolio = pd.DataFrame(columns=['Ticker', 'Qty'])
    
    # Load Alerts
    user_alerts = auth_handler.load_user_alerts(st.session_state.user_id)
    st.session_state.alerts = user_alerts if user_alerts else []
    
    st.session_state.user_loaded = st.session_state.user_id

# PREMIUM CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
    color: #e8eaed;
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    background: linear-gradient(90deg, #00f2ff 0%, #7b2ff7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    letter-spacing: 1px;
}

div[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    background: linear-gradient(135deg, #00f2ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

div[data-testid="stMetricLabel"] {
    color: #9aa0a6;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

section[data-testid="stSidebar"] {
    background: rgba(10, 14, 39, 0.95);
    border-right: 2px solid rgba(0, 242, 255, 0.2);
    backdrop-filter: blur(10px);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(26, 31, 58, 0.6);
    border: 1px solid rgba(123, 47, 247, 0.3);
    border-radius: 12px;
    color: #9aa0a6;
    padding: 12px 24px;
    font-weight: 600;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(123, 47, 247, 0.2);
    border-color: #7b2ff7;
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(123, 47, 247, 0.3);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0, 242, 255, 0.15), rgba(123, 47, 247, 0.15));
    border: 2px solid #00f2ff;
    color: #00f2ff;
    box-shadow: 0 0 20px rgba(0, 242, 255, 0.4);
}

.stButton > button {
    background: linear-gradient(135deg, #00f2ff, #7b2ff7);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 600;
    transition: all 0.3s;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 24px rgba(123, 47, 247, 0.5);
}

.card {
    background: rgba(26, 31, 58, 0.7);
    border: 1px solid rgba(0, 242, 255, 0.2);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(10px);
    transition: all 0.3s;
}

.card:hover {
    border-color: rgba(0, 242, 255, 0.5);
    box-shadow: 0 8px 32px rgba(0, 242, 255, 0.2);
    transform: translateY(-4px);
}

.alert-box {
    background: linear-gradient(135deg, rgba(255, 87, 87, 0.1), rgba(255, 154, 0, 0.1));
    border-left: 4px solid #ff5757;
    padding: 16px;
    border-radius: 8px;
    margin: 12px 0;
}

.chat-bubble {
    background: rgba(0, 242, 255, 0.1);
    border-left: 3px solid #00f2ff;
    padding: 16px;
    border-radius: 8px;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)

# HEADER
col1, col2 = st.columns([3, 1])
with col1:
    st.title("💠 QUANTUM TERMINAL V11 PRO")
with col2:
    st.metric("STATUS", "ONLINE", "✓")

st.markdown(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC+1")
st.markdown("---")

# SESSION STATE INITIALIZATION
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Ticker', 'Qty'])
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# SIDEBAR - ADVANCED ASSET SEARCH & USER PROFILE
st.sidebar.markdown(f"### 👤 {st.session_state.username}")
if st.sidebar.button("🚪 LOGOUT", type="secondary"):
    auth.logout()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 PORTFOLIO MANAGER")

# Search method selector
search_mode = st.sidebar.radio("Search Mode:", ["🔍 Smart Search", "📁 By Category"], horizontal=True)

with st.sidebar.form("add_form"):
    if search_mode == "🔍 Smart Search":
        st.markdown("#### Search Asset")
        search_query = st.text_input("🔎 Search by name or ticker:", placeholder="e.g. Apple, MSFT, Bitcoin...")
        
        selected_ticker = None
        if search_query:
            results = search_assets(search_query)
            if results:
                options = [f"{r['ticker']} - {r['name']} ({r['category']})" for r in results]
                selection = st.selectbox("Select:", options)
                if selection:
                    selected_ticker = selection.split(" - ")[0]
                st.caption(f"✅ Found {len(results)} result(s)")
            else:
                st.warning("No results. Try manual ticker:")
                selected_ticker = st.text_input("Manual Ticker:")
    else:
        # Category-based selection
        st.markdown("#### Browse by Category")
        categories = get_categories()
        selected_category = st.selectbox("Category:", categories)
        assets_in_cat = get_assets_by_category(selected_category)
        
        options = [f"{ticker} - {name}" for ticker, name in assets_in_cat]
        selection = st.selectbox("Asset:", options)
        selected_ticker = selection.split(" - ")[0] if selection else None
    
    qty = st.number_input("Quantity:", min_value=0.1, value=1.0, step=0.1)
    add_btn = st.form_submit_button("➕ ADD TO PORTFOLIO")
    
    if add_btn and selected_ticker:
        new = pd.DataFrame({'Ticker': [selected_ticker], 'Qty': [qty]})
        st.session_state.portfolio = pd.concat([st.session_state.portfolio, new], ignore_index=True)
        
        # SAVE TO FIRESTORE
        auth_handler = auth.FirebaseAuth()
        portfolio_dict = st.session_state.portfolio.to_dict('records')
        auth_handler.save_user_portfolio(st.session_state.user_id, portfolio_dict)
        
        st.rerun()

st.sidebar.markdown("---")

# Active Portfolio Display
if not st.session_state.portfolio.empty:
    st.sidebar.markdown("#### 📊 Active Positions")
    st.sidebar.dataframe(st.session_state.portfolio, hide_index=True, use_container_width=True)
    
    remove = st.sidebar.multiselect("Close:", st.session_state.portfolio['Ticker'].unique())
    if st.sidebar.button("🗑️ REMOVE"):
        st.session_state.portfolio = st.session_state.portfolio[
            ~st.session_state.portfolio['Ticker'].isin(remove)]
            
        # SAVE TO FIRESTORE
        auth_handler = auth.FirebaseAuth()
        portfolio_dict = st.session_state.portfolio.to_dict('records')
        auth_handler.save_user_portfolio(st.session_state.user_id, portfolio_dict)
        
        st.rerun()


# DATA FUNCTIONS
@st.cache_data(ttl=300)
def get_prices(tickers, period="2y"):
    try:
        if isinstance(tickers, str):
            tickers = [tickers]
        data = yf.download(tickers, period=period, progress=False)
        if len(tickers) == 1:
            # Single ticker - return Series
            if 'Adj Close' in data.columns:
                return data['Adj Close']
            elif 'Close' in data.columns:
                return data['Close']
            return data
        else:
            # Multiple tickers
            if 'Adj Close' in data.columns:
                return data['Adj Close']
            elif 'Close' in data.columns:
                return data['Close']
            return data
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

@st.cache_data(ttl=300)
def get_asset_info(ticker):
    """Get asset fundamentals safely"""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        if info is None:
            return {}, 'UNKNOWN'
        
        qType = info.get('quoteType', 'UNKNOWN').upper()
        
        # Override for futures
        if "=F" in ticker or "BTP" in ticker:
            qType = 'FUTURE'
        
        # Normalize type
        if 'ETF' in qType:
            rType = 'ETF'
        elif 'CRYPTOCURRENCY' in qType:
            rType = 'CRYPTO'
        elif 'FUTURE' in qType:
            rType = 'FUTURE'
        elif 'INDEX' in qType:
            rType = 'INDEX'
        else:
            rType = 'EQUITY'
        
        return info, rType
    except:
        return {}, 'UNKNOWN'

def calc_rsi(series, period=14):
    try:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except:
        return pd.Series([50]*len(series))

def get_signal(price, sma50, sma200, rsi):
    score = 0
    reasons = []
    
    if pd.isna(price) or pd.isna(sma50) or pd.isna(sma200):
        return "⚪ NO DATA", [], "#8b949e"
    
    if price > sma200:
        score += 2
        reasons.append("Price > SMA200")
    else:
        score -= 2
        reasons.append("Price < SMA200")
    
    if price > sma50:
        score += 1
    else:
        score -= 1
    
    if rsi < 30:
        score += 2
        reasons.append(f"RSI Oversold ({rsi:.0f})")
    elif rsi > 70:
        score -= 2
        reasons.append(f"RSI Overbought ({rsi:.0f})")
    
    if score >= 2:
        return "🟢 BUY", reasons, "#39d353"
    elif score <= -2:
        return "🔴 SELL", reasons, "#f85149"
    return "🟡 HOLD", reasons, "#e3b341"

@st.cache_data(ttl=600)
def calculate_hurst(series, min_window=10):
    """
    Calculate Hurst Exponent (H) for a time series
    HO > 0.5 = Persistent (Trending)
    H = 0.5 = Random Walk
    H < 0.5 = Mean Reverting
    """
    try:
        # Simplified R/S Analysis
        lags = range(2, 20)
        tau = [np.sqrt(np.std(series.diff(lag).dropna())) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2  # Approximate relation
    except:
        return 0.5

def calculate_shannon_entropy(series, bins=20):
    """Calculate Shannon Entropy of return distribution"""
    try:
        returns = series.pct_change().dropna()
        hist, bin_edges = np.histogram(returns, bins=bins, density=True)
        # Add small constant to avoid log(0)
        hist = hist[hist > 0]
        return entropy(hist)
    except:
        return 0

def calculate_pareto_alpha(series):
    """
    Calculate Pareto Exponent (Alpha) for tail risk.
    Alpha < 2 implies infinite variance (heavy tails).
    """
    try:
        returns = series.pct_change().dropna().abs()
        # Focus on the tail (top 20%)
        threshold = returns.quantile(0.80)
        tail = returns[returns > threshold]
        
        # Hill Estimator
        n = len(tail)
        if n == 0: return 0
        
        log_sum = np.sum(np.log(tail / threshold))
        alpha = n / log_sum
        return alpha
    except:
        return 0

def calculate_multifractal_spectrum(series):
    """
    Calculate Multifractal Spectrum Width (Delta Alpha).
    Wider spectrum = Higher Multifractality = Higher complexity/risk.
    """
    try:
        # Simplified MFDFA (Multifractal Detrended Fluctuation Analysis) proxy
        # We calculate Hurst exponents for different q-orders (small vs large fluctuations)
        
        returns = series.pct_change().dropna()
        window_sizes = [10, 20, 40, 80]
        q_orders = [-2, 2] # Small fluctuations vs Large fluctuations
        
        hursts = []
        for q in q_orders:
            fluctuations = []
            for w in window_sizes:
                # Calculate fluctuation function F_q(w)
                rolling_std = returns.rolling(w).std().dropna()
                f_q = np.mean(rolling_std ** q) ** (1/q)
                fluctuations.append(f_q)
            
            # Fit log-log to get H(q)
            if len(fluctuations) == len(window_sizes) and all(f > 0 for f in fluctuations):
                coeffs = np.polyfit(np.log(window_sizes), np.log(fluctuations), 1)
                hursts.append(coeffs[0])
        
        if len(hursts) == 2:
            # Delta Alpha approx = H(-2) - H(2)
            # Positive spread indicates multifractality
            return abs(hursts[0] - hursts[1])
        return 0
    except:
        return 0

def generate_trade_network():
    """
    Simulate Global Trade Network for PageRank Visualization
    Nodes = Countries, Edges = Trade Volume
    """
    countries = ["USA", "China", "Germany", "Japan", "India", "UK", "France", "Brazil", "Russia", "Saudi Arabia"]
    
    # Create random but weighted graph (China/USA hub-like)
    G = nx.DiGraph()
    
    # Trade flows (simulated weight)
    edges = [
        ("China", "USA", 550), ("China", "Germany", 200), ("China", "Japan", 150),
        ("USA", "China", 150), ("USA", "Germany", 180), ("USA", "UK", 120),
        ("Germany", "France", 180), ("Germany", "China", 120), ("Germany", "USA", 160),
        ("Saudi Arabia", "USA", 80), ("Saudi Arabia", "China", 100), ("Saudi Arabia", "Japan", 90),
        ("Russia", "China", 110), ("Russia", "India", 90),
        ("India", "USA", 100), ("India", "China", 80),
        ("Brazil", "China", 120), ("Brazil", "USA", 80)
    ]
    
    G.add_weighted_edges_from(edges)
    
    # Calculate PageRank
    pagerank = nx.pagerank(G, weight='weight')
    
    return G, pagerank

def calculate_dcca(series1, series2, window=40):
    """
    Calculate Dynamic Cross-Correlation (DCCA) coefficient over time.
    Uses rolling window with local detrending.
    """
    try:
        # Align series
        df = pd.concat([series1, series2], axis=1).dropna()
        if len(df) < window:
            return pd.Series()
            
        s1 = df.iloc[:, 0]
        s2 = df.iloc[:, 1]
        
        dcca_rho = []
        indices = []
        
        # Sliding window
        for i in range(window, len(df)):
            w1 = s1.iloc[i-window:i].values
            w2 = s2.iloc[i-window:i].values
            
            # Local Detrending (Linear)
            x = np.arange(window)
            
            # Fit and remove trend S1
            p1 = np.polyfit(x, w1, 1)
            trend1 = np.polyval(p1, x)
            resid1 = w1 - trend1
            
            # Fit and remove trend S2
            p2 = np.polyfit(x, w2, 1)
            trend2 = np.polyval(p2, x)
            resid2 = w2 - trend2
            
            # Covariance of residuals
            cov = np.mean(resid1 * resid2)
            var1 = np.mean(resid1**2)
            var2 = np.mean(resid2**2)
            
            if var1 > 0 and var2 > 0:
                rho = cov / np.sqrt(var1 * var2)
            else:
                rho = 0
                
            dcca_rho.append(rho)
            indices.append(df.index[i])
            
        return pd.Series(dcca_rho, index=indices)
    except:
        return pd.Series()

def rmt_clean_covariance(returns):
    """
    Denoise covariance matrix using Random Matrix Theory (Marcenko-Pastur).
    Removes noise eigenvalues essentially filtering out random correlations.
    """
    try:
        corr = returns.corr()
        cov = returns.cov()
        
        # Eigen decomposition
        vals, vecs = np.linalg.eigh(corr)
        
        # Marcenko-Pastur Threshold
        T, N = returns.shape
        Q = T / N
        sigma = 1 # variance of standardized variables
        lambda_max = sigma**2 * (1 + (1/Q) + 2*np.sqrt(1/Q))
        
        # Filter eigenvalues
        vals_clean = vals.copy()
        vals_clean[vals < lambda_max] = np.mean(vals[vals < lambda_max]) # Replace noise with average
        
        # Reconstruct correlation matrix
        corr_clean = np.dot(vecs, np.dot(np.diag(vals_clean), vecs.T))
        np.fill_diagonal(corr_clean, 1)
        
        # Convert back to covariance
        std = np.sqrt(np.diag(cov))
        cov_clean = np.outer(std, std) * corr_clean
        
        return pd.DataFrame(cov_clean, index=cov.index, columns=cov.columns)
    except:
        return returns.cov()

def calculate_hrp_weights(returns):
    """
    Hierarchical Risk Parity (HRP) Optimization.
    1. Clustering (Linkage)
    2. Quasi-Diagonalization
    3. Recursive Bisection Division
    """
    try:
        from scipy.cluster.hierarchy import linkage, to_tree
        from scipy.spatial.distance import squareform

        corr = returns.corr()
        cov = returns.cov()
        
        # 1. Clustering
        dist = np.sqrt((1 - corr) / 2)
        link = linkage(squareform(dist), 'single')
        
        # 2. Quasi-Diagonalization (Sort indices)
        def get_quasi_diag(link):
            link = link.astype(int)
            sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
            num_items = link[-1, 3]
            while sort_ix.max() >= num_items:
                sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
                df0 = sort_ix[sort_ix >= num_items]
                i = df0.index
                j = df0.values - num_items
                sort_ix[i] = link[j, 0]
                df0 = pd.Series(link[j, 1], index=i + 1)
                sort_ix = pd.concat([sort_ix, df0]) #.append(df0)
                sort_ix = sort_ix.sort_index()
                sort_ix.index = range(sort_ix.shape[0])
            return sort_ix.tolist()
            
        sort_ix = get_quasi_diag(link)
        sort_ix = corr.index[sort_ix].tolist()
        
        df_sorted = cov.loc[sort_ix, sort_ix]
        
        # 3. Recursive Bisection
        weights = pd.Series(1, index=sort_ix)
        
        def get_cluster_var(cov, c_items):
            cov_slice = cov.loc[c_items, c_items]
            w = 1 / np.diag(cov_slice) # Inverse variance weights for sub-cluster
            w = w / w.sum()
            return np.dot(np.dot(w, cov_slice), w)
            
        def recurse_bisection(w, cov, items):
            if len(items) > 1:
                split = len(items) // 2
                left = items[:split]
                right = items[split:]
                
                v_left = get_cluster_var(cov, left)
                v_right = get_cluster_var(cov, right)
                
                alpha = 1 - v_left / (v_left + v_right)
                
                w[left] *= alpha
                w[right] *= (1 - alpha)
                
                recurse_bisection(w, cov, left)
                recurse_bisection(w, cov, right)
        
        recurse_bisection(weights, df_sorted, sort_ix)
        return weights
    except Exception as e:
        # Fallback to equal weight
        return pd.Series(1/len(returns.columns), index=returns.columns)

def calculate_vasicek_params(series):
    """
    Calibrate Vasicek Mean Reversion Model: dr = a(b - r)dt + sigma*dW
    Returns: Mean Level (b), Speed of Reversion (a), Volatility (sigma)
    """
    try:
        # Works best on Yields (e.g., ^TNX), but can apply OU process to mean-reverting prices
        x = series[:-1].values
        y = series[1:].values
        
        # Regression: r(t+1) = alpha + beta * r(t) + epsilon
        # slope (beta) = 1 - a*dt
        # intercept (alpha) = a*b*dt
        
        coeffs = np.polyfit(x, y, 1)
        beta = coeffs[0]
        alpha = coeffs[1]
        
        dt = 1/252 # Daily steps
        
        a = (1 - beta) / dt
        b = alpha / (1 - beta)
        
        residuals = y - (alpha + beta * x)
        sigma = np.std(residuals) / np.sqrt(dt)
        
        return a, b, sigma
    except:
        return 0, 0, 0

def calculate_cir_params(series):
    """
    Calibrate CIR Model: dr = a(b - r)dt + sigma*sqrt(r)*dW
    Model enforces non-negativity (good for rates).
    """
    try:
        # Discrete approximation regression
        # (r(t+1) - r(t)) / sqrt(r(t)) = a*b*dt/sqrt(r(t)) - a*dt*sqrt(r(t)) + sigma*epsilon
        
        r_t = series[:-1].values
        dr = np.diff(series.values)
        dt = 1/252
        
        y = dr / np.sqrt(np.abs(r_t)) # abs to handle negative yields if any (though CIR assumes +)
        x1 = dt / np.sqrt(np.abs(r_t))
        x2 = dt * np.sqrt(np.abs(r_t))
        
        # Multiple linear regression is complex here manually, let's use simplified OLS on discretized version
        # r(t+1) = r(t) + a(b-r(t))dt
        # This is same drift as Vasicek, just volatility structure differs for sim. 
        # For param estimation 'a' and 'b', we can use Vasicek estimates as proxy or simple regression
        # We'll use the Vasicek 'a' and 'b' for drift, and estimate sigma from scaled residuals
        
        a, b, _ = calculate_vasicek_params(series)
        
        # Estimate sigma for CIR
        # dr = Drift + sigma * sqrt(r) * dW
        # sigma = std(dr - Drift) / sqrt(r*dt)
        
        drift = a * (b - r_t) * dt
        residuals = dr - drift
        
        # Average sigma estimate
        sigma_sq = np.mean(residuals**2 / (np.abs(r_t) * dt))
        sigma = np.sqrt(sigma_sq)
        
        return a, b, sigma
    except:
        return 0, 0, 0

def get_sector_clusters():
    """Simulate Sector Clustering Data"""
    return {
        "AI & Semi": {"Nvidia": 0.95, "AMD": 0.88, "TSMC": 0.85, "ASML": 0.82},
        "Quantum": {"IonQ": 0.75, "Rigetti": 0.72, "D-Wave": 0.68},
        "Defense": {"Lockheed": 0.45, "Raytheon": 0.48, "Northrop": 0.42},
        "Crypto": {"Bitcoin": 0.92, "Ethereum": 0.90, "Solana": 0.85}
    }

def get_news():
    """Get financial news from Google News RSS"""
    try:
        feed = feedparser.parse("https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en")
        return feed.entries[:8]
    except:
        return []

def calculate_smartquant_score(ticker, series, info, asset_type):
    """
    Calculate Q-Score Score (1-100) similar to Quantalys
    Combines Technical + Fundamental + Momentum + Quality indicators
    """
    try:
        score = 50  # Base score
        signals = []
        
        if len(series) < 50:
            return 50, "NEUTRAL", signals, "#e3b341"
        
        curr_price = series.iloc[-1]
        
        # === TECHNICAL ANALYSIS (40 points) ===
        
        # 1. Trend Analysis (20 pts)
        sma50 = series.rolling(50).mean().iloc[-1] if len(series) >= 50 else curr_price
        sma200 = series.rolling(200).mean().iloc[-1] if len(series) >= 200 else curr_price
        
        if curr_price > sma200:
            if curr_price > sma50:
                score += 20
                signals.append("📈 Strong uptrend (above SMA50 & SMA200)")
            else:
                score += 10
                signals.append("📊 Above SMA200 but below SMA50")
        else:
            if curr_price < sma50:
                score -= 20
                signals.append("📉 Strong downtrend (below both SMAs)")
            else:
                score -= 10
                signals.append("⚠️ Below SMA200")
        
        # 2. Momentum - RSI (10 pts)
        rsi = calc_rsi(series).iloc[-1]
        if 40 < rsi < 60:
            score += 10
            signals.append(f"✅ Healthy RSI ({rsi:.0f})")
        elif rsi < 30:
            score += 5
            signals.append(f"💎 Oversold RSI ({rsi:.0f}) - potential buy")
        elif rsi > 70:
            score -= 10
            signals.append(f"⚠️ Overbought RSI ({rsi:.0f})")
        
        # 3. Recent Performance (10 pts)
        if len(series) >= 21:
            month_ret = ((curr_price - series.iloc[-21]) / series.iloc[-21]) * 100
            if month_ret > 5:
                score += 10
                signals.append(f"🚀 Strong momentum (+{month_ret:.1f}% month)")
            elif month_ret > 0:
                score += 5
                signals.append(f"📈 Positive momentum (+{month_ret:.1f}% month)")
            elif month_ret < -10:
                score -= 10
                signals.append(f"📉 Weak momentum ({month_ret:.1f}% month)")
        
        # === FUNDAMENTAL ANALYSIS (30 points) - Only for Stocks ===
        if asset_type == 'EQUITY' and info:
            # 4. Valuation - P/E (15 pts)
            pe = info.get('trailingPE', info.get('forwardPE', None))
            if pe:
                if 10 < pe < 20:
                    score += 15
                    signals.append(f"💰 Attractive valuation (P/E {pe:.1f})")
                elif 20 <= pe < 30:
                    score += 8
                    signals.append(f"📊 Fair valuation (P/E {pe:.1f})")
                elif pe >= 40:
                    score -= 10
                    signals.append(f"⚠️ High valuation (P/E {pe:.1f})")
            
            # 5. Profitability - Profit Margin (15 pts)
            profit_margin = info.get('profitMargins', None)
            if profit_margin:
                if profit_margin > 0.20:
                    score += 15
                    signals.append(f"💎 Excellent margins ({profit_margin*100:.1f}%)")
                elif profit_margin > 0.10:
                    score += 8
                    signals.append(f"✅ Good margins ({profit_margin*100:.1f}%)")
                elif profit_margin < 0:
                    score -= 15
                    signals.append("⚠️ Unprofitable")
        
        # === VOLATILITY & RISK (20 points) ===
        # 6. Volatility Analysis
        if len(series) >= 30:
            returns = series.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            
            if volatility < 20:
                score += 10
                signals.append(f"🛡️ Low volatility ({volatility:.1f}%)")
            elif volatility < 40:
                score += 5
                signals.append(f"📊 Moderate volatility ({volatility:.1f}%)")
            else:
                score -= 5
                signals.append(f"⚡ High volatility ({volatility:.1f}%)")
        
        # === ETF/CRYPTO SPECIFIC ===
        if asset_type == 'ETF':
            expense_ratio = info.get('annualReportExpenseRatio', info.get('expenseRatio', 0))
            if expense_ratio and expense_ratio < 0.005:
                score += 10
                signals.append(f"💰 Low fees ({expense_ratio*100:.2f}%)")
        
        if asset_type == 'CRYPTO':
            # Crypto bonus for market cap
            mcap = info.get('marketCap', 0)
            if mcap > 100e9:
                score += 10
                signals.append("🏆 Large cap crypto (more stable)")
            elif mcap < 1e9:
                score -= 10
                signals.append("⚠️ Small cap crypto (high risk)")
        
        # Clamp score between 0-100
        score = max(0, min(100, score))
        
        # Determine rating
        if score >= 80:
            rating = "STRONG BUY"
            color = "#00ff00"
        elif score >= 65:
            rating = "BUY"
            color = "#39d353"
        elif score >= 45:
            rating = "HOLD"
            color = "#e3b341"
        elif score >= 30:
            rating = "SELL"
            color = "#f85149"
        else:
            rating = "STRONG SELL"
            color = "#ff0000"
        
        return score, rating, signals, color
        
    except Exception as e:
        return 50, "NEUTRAL", [f"Error calculating score: {str(e)}"], "#8b949e"


# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "� PORTFOLIO OVERVIEW", "🧬 OPTIMIZATION", "🌍 MACRO", "🔔 ALERTS", "🤖 AI ASSISTANT"
])

# TAB 1: PORTFOLIO OVERVIEW WITH SMARTQUANT
with tab1:
    st.subheader("� Portfolio Overview & Q-Score Scores")
    
    if not st.session_state.portfolio.empty:
        tickers = st.session_state.portfolio['Ticker'].unique().tolist()
        
        # Calculate total value for allocation %
        total_qty = st.session_state.portfolio['Qty'].sum()
        
        # Create summary data
        st.markdown("### 💼 Portfolio Summary")
        
        summary_data = []
        
        with st.spinner("📊 Analyzing portfolio..."):
            for ticker in tickers:
                try:
                    # Get data
                    series = get_prices(ticker, period="1y")
                    if series is None or len(series) < 50:
                        continue
                    
                    if isinstance(series, pd.DataFrame):
                        series = series.iloc[:, 0]
                    series = series.dropna()
                    
                    info, asset_type = get_asset_info(ticker)
                    curr_price = series.iloc[-1]
                    
                    # Calculate Q-Score Score
                    sq_score, sq_rating, sq_signals, sq_color = calculate_smartquant_score(
                        ticker, series, info, asset_type
                    )
                    
                    # Calculate Physics Metrics
                    hurst = calculate_hurst(series)
                    pareto = calculate_pareto_alpha(series)
                    mf_width = calculate_multifractal_spectrum(series)
                    
                    # Get quantity and allocation
                    qty = st.session_state.portfolio[
                        st.session_state.portfolio['Ticker'] == ticker
                    ]['Qty'].iloc[0]
                    
                    allocation = (qty / total_qty) * 100
                    
                    # Price changes
                    day_change = ((curr_price - series.iloc[-2]) / series.iloc[-2]) * 100 if len(series) >= 2 else 0
                    week_change = ((curr_price - series.iloc[-5]) / series.iloc[-5]) * 100 if len(series) >= 5 else 0
                    
                    summary_data.append({
                        'ticker': ticker,
                        'name': info.get('longName', ticker) if info else ticker,
                        'type': asset_type,
                        'price': curr_price,
                        'day_change': day_change,
                        'week_change': week_change,
                        'sq_score': sq_score,
                        'sq_rating': sq_rating,
                        'sq_color': sq_color,
                        'sq_signals': sq_signals,
                        'allocation': allocation,
                        'qty': qty,
                        'series': series,
                        'info': info,
                        'hurst': hurst,
                        'pareto': pareto,
                        'mf_width': mf_width
                    })
                    
                except Exception as e:
                    st.error(f"Error processing {ticker}: {str(e)}")
                    continue
        
        if summary_data:
            # COMPACT TABLE VIEW
            st.markdown("#### 📋 Quick View - All Positions")
            
            # Create DataFrame for display
            table_data = []
            for asset in summary_data:
                table_data.append({
                    'Ticker': asset['ticker'],
                    'Asset': asset['name'][:30] + '...' if len(asset['name']) > 30 else asset['name'],
                    'Type': asset['type'],
                    'Price': f"${asset['price']:,.2f}",
                    'Day %': f"{asset['day_change']:+.2f}%",
                    'Week %': f"{asset['week_change']:+.2f}%",
                    '🎯 Q-Score': f"{asset['sq_score']}/100",
                    'Rating': asset['sq_rating'],
                    'Allocation': f"{asset['allocation']:.1f}%",
                    'Hurst (H)': f"{asset['hurst']:.2f}",
                    'Pareto (α)': f"{asset['pareto']:.1f}"
                })
            
            df_display = pd.DataFrame(table_data)
            
            # Color code the ratings
            def color_rating(val):
                if 'STRONG BUY' in val:
                    return 'background-color: rgba(0, 255, 0, 0.2); color: #00ff00; font-weight: bold'
                elif 'BUY' in val:
                    return 'background-color: rgba(57, 211, 83, 0.2); color: #39d353; font-weight: bold'
                elif 'HOLD' in val:
                    return 'background-color: rgba(227, 179, 65, 0.2); color: #e3b341; font-weight: bold'
                elif 'SELL' in val:
                    return 'background-color: rgba(248, 81, 73, 0.2); color: #f85149; font-weight: bold'
                return ''
            
            styled_df = df_display.style.applymap(color_rating, subset=['Rating'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # DETAILED EXPANDABLE VIEW
            st.markdown("#### 🔍 Detailed Analysis - Click to Expand")
            
            for asset in summary_data:
                with st.expander(f"📈 {asset['ticker']} - {asset['name']} | Q-Score: **{asset['sq_score']}/100** ({asset['sq_rating']})", expanded=False):
                    
                    # Q-Score Score Card
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(0, 242, 255, 0.1), rgba(123, 47, 247, 0.1)); 
                                border: 2px solid {asset['sq_color']}; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                        <h3 style="color: {asset['sq_color']}; margin: 0;">
                            🎯 Q-Score Score: {asset['sq_score']}/100
                        </h3>
                        <h4 style="color: {asset['sq_color']}; margin: 10px 0 0 0;">
                            Raccomandazione: {asset['sq_rating']}
                        </h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics Row 1: Basic
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("💰 Price", f"${asset['price']:,.2f}", f"{asset['day_change']:+.2f}%")
                    c2.metric("📅 Week", f"{asset['week_change']:+.2f}%")
                    c3.metric("📊 Allocation", f"{asset['allocation']:.1f}%")
                    c4.metric("🎲 Quantity", f"{asset['qty']}")
                    c5.metric("📂 Type", asset['type'])
                    
                    # Metrics Row 2: Physics
                    st.markdown("##### ⚛️ Asset Physics & Complexity")
                    p1, p2, p3 = st.columns(3)
                    
                    p1.metric("Hurst Exponent (H)", f"{asset['hurst']:.2f}", 
                             "Trending 🚀" if asset['hurst'] > 0.6 else "Mean Reverting 🔄" if asset['hurst'] < 0.4 else "Random 🎲")
                    p2.metric("Pareto Alpha (α)", f"{asset['pareto']:.2f}", "Fat Tail Risk ⚠️" if asset['pareto'] < 2 else "Normal Tail ✅")
                    p3.metric("Multifractal Spectrum (Δα)", f"{asset['mf_width']:.3f}", 
                             "High Complexity 🌀" if asset['mf_width'] > 0.3 else "Low Complexity 🧊")
                    
                    # Q-Score Signals
                    st.markdown("##### 🔍 Analysis Factors:")
                    for signal in asset['sq_signals']:
                        st.markdown(f"- {signal}")
                    
                    # Price Chart
                    st.markdown("##### 📉 Price Chart (90 Days)")
                    recent = asset['series'].iloc[-90:].to_frame('Price')
                    recent['SMA20'] = asset['series'].rolling(20).mean().iloc[-90:]
                    recent['SMA50'] = asset['series'].rolling(50).mean().iloc[-90:]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=recent.index, y=recent['Price'],
                        mode='lines', name='Price',
                        line=dict(color='#00f2ff', width=2.5)
                    ))
                    fig.add_trace(go.Scatter(
                        x=recent.index, y=recent['SMA20'],
                        mode='lines', name='SMA 20',
                        line=dict(color='#7b2ff7', width=1.5, dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        x=recent.index, y=recent['SMA50'],
                        mode='lines', name='SMA 50',
                        line=dict(color='#39d353', width=1.5, dash='dot')
                    ))
                    
                    fig.update_layout(
                        template="plotly_dark",
                        height=300,
                        margin=dict(l=0, r=0, t=20, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(26,31,58,0.5)',
                        hovermode='x unified',
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Fundamentals (if available)
                    if asset['info']:
                        st.markdown("##### 📋 Fundamentals")
                        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
                        
                        info = asset['info']
                        if asset['type'] == 'EQUITY':
                            pe = info.get('trailingPE', info.get('forwardPE', None))
                            mcap = info.get('marketCap', 0)
                            div_yield = info.get('dividendYield', 0)
                            margin = info.get('profitMargins', 0)
                            
                            fcol1.metric("P/E", f"{pe:.2f}" if pe else "N/A")
                            fcol2.metric("Market Cap", f"${mcap/1e9:.2f}B" if mcap else "N/A")
                            fcol3.metric("Div Yield", f"{div_yield*100:.2f}%" if div_yield else "N/A")
                            fcol4.metric("Profit Margin", f"{margin*100:.1f}%" if margin else "N/A")
                        
                        elif asset['type'] == 'ETF':
                            yield_val = info.get('yield', info.get('trailingAnnualDividendYield', 0))
                            assets_val = info.get('totalAssets', 0)
                            expense = info.get('annualReportExpenseRatio', info.get('expenseRatio', 0))
                            
                            fcol1.metric("Yield", f"{yield_val*100:.2f}%" if yield_val else "N/A")
                            fcol2.metric("Assets", f"${assets_val/1e9:.2f}B" if assets_val else "N/A")
                            fcol3.metric("Expense Ratio", f"{expense*100:.2f}%" if expense else "N/A")
                            fcol4.metric("Type", "ETF")

                        # Special Section for Bonds/Yields
                        if "Treasury" in asset['name'] or "Bond" in asset['name'] or asset['type'] == 'BOND' or ticker in ["^TNX", "^TYX", "TLT", "IEF", "SHY"]:
                            st.markdown("##### 📜 Bond Physics (Stochastic Models)")
                            
                            # Calibrate Models
                            va_a, va_b, va_sigma = calculate_vasicek_params(asset['series'])
                            cir_a, cir_b, cir_sigma = calculate_cir_params(asset['series'])
                            
                            b1, b2, b3 = st.columns(3)
                            
                            b1.metric("Short Rate (Current)", f"{asset['series'].iloc[-1]:.2f}%" if ticker.startswith("^") else f"${asset['series'].iloc[-1]:.2f}")
                            
                            b2.metric("Vasicek Mean (θ)", f"{va_b:.2f}", 
                                     help="Long-term mean level the rate/price reverts to.")
                            
                            b3.metric("Reversion Speed (κ)", f"{va_a:.2f}", 
                                     help="How fast it returns to the mean. Higher = Stronger magnet.")
                            
                            st.caption(f"**CIR Volatility:** {cir_sigma:.2%} | **Vasicek Volatility:** {va_sigma:.2%}")
                            
                            if ticker.startswith("^"):
                                st.info("ℹ️ Models calibrated on Yield Data (Interpreted as Rates).")
                            else:
                                st.info("ℹ️ Models calibrated on Price Data (Interpreted as Mean-Reverting Price Process).")
                        
                        elif asset['type'] == 'CRYPTO':
                            mcap = info.get('marketCap', 0)
                            volume = info.get('volume24Hr', info.get('volume', 0))
                            
                            fcol1.metric("Market Cap", f"${mcap/1e9:.2f}B" if mcap else "N/A")
                            fcol2.metric("24h Volume", f"${volume/1e9:.2f}B" if volume else "N/A")
                            fcol3.metric("Type", "Cryptocurrency")
        else:
            st.warning("⚠️ Could not fetch data for portfolio assets")
            
    else:
        st.info("💡 **Add assets from the sidebar to see your portfolio overview**")
        st.markdown("""
        ### 🎯 Features:
        - **Q-Score Score** - AI-powered rating system (0-100)
        - **Allocation View** - See portfolio weights
        - **Compact Table** - All positions at a glance
        - **Detailed Expanders** - Click to see full analysis
        - **Live Ratings** - STRONG BUY, BUY, HOLD, SELL, STRONG SELL
        
        **Add 1+ assets to get started!** 🚀
        """)



# TAB 2: PORTFOLIO
with tab2:
    st.subheader("🧬 Quantum Optimization Studio")
    
    if not st.session_state.portfolio.empty:
        tickers = st.session_state.portfolio['Ticker'].unique().tolist()
        
        if len(tickers) >= 2:
            # Inputs
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            with col_opt1:
                risk_level = st.slider("⚖️ Risk Tolerance (1-10)", 1, 10, 5, help="1=Conservative, 10=Aggressive")
            with col_opt2:
                capital = st.number_input("💰 Investment Capital ($)", value=10000, step=1000)
            with col_opt3:
                model_choice = st.selectbox("🧮 Optimization Model", 
                                          ["Markowitz (Classical)", 
                                           "Quantum RMT (Denoised)", 
                                           "Hierarchical Risk Parity (HRP)", 
                                           "Black-Litterman (AI Views)"])
            
            # Fetch price data for all assets
            with st.spinner("📊 Downloading & Processing Quantum Data..."):
                portfolio_prices = get_prices(tickers, period="2y")
            
            if portfolio_prices is not None:
                # Handle single vs multiple tickers
                if isinstance(portfolio_prices, pd.Series):
                    portfolio_prices = portfolio_prices.to_frame(name=tickers[0])
                
                # Clean data - remove columns with insufficient data
                portfolio_prices = portfolio_prices.dropna(axis=1, thresh=len(portfolio_prices)*0.7)
                portfolio_prices = portfolio_prices.dropna()
                
                if len(portfolio_prices.columns) < 2:
                    st.warning("⚠️ Need at least 2 assets with valid data for optimization")
                elif len(portfolio_prices) < 50:
                    st.warning("⚠️ Insufficient historical data for reliable optimization (need 50+ days)")
                else:
                    # Calculate returns
                    returns = portfolio_prices.pct_change().dropna()
                    
                    # --- MODEL LOGIC ---
                    
                    # 1. COVARIANCE MATRIX
                    if model_choice == "Quantum RMT (Denoised)":
                        cov_matrix = rmt_clean_covariance(returns) * 252
                        st.caption("✨ Covariance matrix denoised using Marcenko-Pastur Law (Random Matrix Theory).")
                    else:
                        cov_matrix = returns.cov() * 252

                    # 2. EXPECTED RETURNS
                    if model_choice == "Black-Litterman (AI Views)":
                        # Simplistic BL: Adjust mean returns based on Q-Score
                        current_means = returns.mean() * 252
                        adjusted_means = []
                        
                        views_text = []
                        for t in returns.columns:
                            # We need to fetch info again for Q-Score or cache it. 
                            # For speed, we'll use a randomized proxy or simple SMA trend as 'View' if strictly needed,
                            # but better to fetch Q-Score if possible. 
                            # Let's use SMA trend as "AI View" proxy for speed here to avoid 10 API calls.
                            trend = calc_rsi(portfolio_prices[t]).iloc[-1]
                            # View: If RSI < 30 (Oversold) -> Expect higher return. If RSI > 70 -> Lower.
                            
                            view_strength = (50 - trend) / 100 # +0.2 to -0.2
                            adj_ret = current_means[t] * (1 + view_strength)
                            adjusted_means.append(adj_ret)
                            views_text.append(f"{t}: {current_means[t]:.2%} -> {adj_ret:.2%} (RSI: {trend:.0f})")
                            
                        expected_returns = pd.Series(adjusted_means, index=returns.columns)
                        st.caption("🧠 Expected Returns adjusted by AI Views (RSI/Trend Factors).")
                        with st.expander("Show AI Views Details"):
                            st.write(views_text)
                    else:
                        expected_returns = returns.mean() * 252
                    
                    # 3. WEIGHT OPTIMIZATION
                    
                    if model_choice == "Hierarchical Risk Parity (HRP)":
                        # HRP doesn't use expected returns, only covariance structure
                        optimal_weights = calculate_hrp_weights(returns)
                        st.caption("🌲 Weights determined by Hierarchical Clustering (No inversion of covariance matrix needed).")
                        
                        # Calculate portfolio stats for HRP
                        port_ret = np.dot(optimal_weights, expected_returns)
                        port_vol = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
                        port_sharpe = port_ret / port_vol
                        
                    else:
                        # Mean-Variance Optimization (Markowitz / RMT / BL)
                        # Monte Carlo Simulation
                        n_portfolios = 5000
                        weights_arr = np.random.random((n_portfolios, len(returns.columns)))
                        weights_arr = (weights_arr.T / weights_arr.sum(axis=1)).T
                        
                        # Portfolio stats
                        p_ret = np.dot(weights_arr, expected_returns)
                        p_vol = np.sqrt(np.einsum('ij,ji->i', np.dot(weights_arr, cov_matrix), weights_arr.T))
                        p_sharpe = p_ret / p_vol
                        
                        # Filter by Risk Level (Target Volatility)
                        # Risk 1 = Min Vol, Risk 10 = Max Sharpe/Max Vol
                        if risk_level == 1:
                            best_idx = p_vol.argmin()
                        elif risk_level == 10:
                            best_idx = p_sharpe.argmax()
                        else:
                            # Interpolate
                            sorted_indices = np.argsort(p_vol)
                            target_idx = int(len(sorted_indices) * (risk_level/10))
                            # Pick best sharpe within that risk tier
                            subset = sorted_indices[max(0, target_idx-500):min(n_portfolios, target_idx+500)]
                            if len(subset) > 0:
                                best_subset_idx = p_sharpe[subset].argmax()
                                best_idx = subset[best_subset_idx]
                            else:
                                best_idx = p_sharpe.argmax()
                        
                        optimal_weights = pd.Series(weights_arr[best_idx], index=returns.columns)
                        port_ret = p_ret[best_idx]
                        port_vol = p_vol[best_idx]
                        port_sharpe = p_sharpe[best_idx]
                        
                        # Plot Efficient Frontier
                        sim_df = pd.DataFrame({'Volatility': p_vol, 'Return': p_ret, 'Sharpe': p_sharpe})
                        fig = px.scatter(sim_df, x='Volatility', y='Return', color='Sharpe', 
                                       title=f"Efficient Frontier ({model_choice})", color_continuous_scale='Plasma')
                        fig.add_trace(go.Scatter(x=[port_vol], y=[port_ret], mode='markers', 
                                                marker=dict(size=20, color='#00f2ff', symbol='star'), name='Selected Portfolio'))
                        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,31,58,0.5)')
                        st.plotly_chart(fig, use_container_width=True)

                    # --- RESULTS DISPLAY ---
                    
                    st.markdown(f"#### 🏆 Optimal Allocation ({model_choice})")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("📈 Expected Return", f"{port_ret*100:.2f}%")
                    col2.metric("📊 Volatility (Risk)", f"{port_vol*100:.2f}%")
                    col3.metric("⚡ Sharpe Ratio", f"{port_sharpe:.2f}")
                    
                    # Liquidity VaR Calculation (Placeholder Proxy)
                    # L-VaR = VaR + (Cost to Liquidation)
                    # We assume 1% liquidation cost for calculation
                    var_95 = 1.65 * port_vol # 95% Confidence
                    lvar = var_95 + 0.01 # +1% Liquidity spread buffer
                    col4.metric("💧 Liquidity VaR (95%)", f"{lvar*100:.2f}%", help="Value at Risk adjusted for Liquidity constraints")
                    
                    st.markdown("##### 💼 Recommended Weights")
                    
                    weights_df = pd.DataFrame({
                        'Asset': optimal_weights.index,
                        'Weight (%)': optimal_weights.values * 100,
                        'Value ($)': optimal_weights.values * capital,
                        'Allocation': ['🟢' * int(w*20) for w in optimal_weights.values]
                    }).sort_values('Weight (%)', ascending=False)
                    
                    # Filter small weights
                    weights_df_display = weights_df[weights_df['Weight (%)'] > 0.5].copy()
                    
                    # Formatting
                    weights_df_display['Weight (%)'] = weights_df_display['Weight (%)'].apply(lambda x: f"{x:.2f}%")
                    weights_df_display['Value ($)'] = weights_df_display['Value ($)'].apply(lambda x: f"${x:,.2f}")
                    
                    st.dataframe(weights_df_display, hide_index=True, use_container_width=True)
                    
                    # Pie Chart of Allocation
                    fig_pie = px.pie(weights_df[weights_df['Weight (%)'] > 0.5], values='Value ($)', names='Asset', 
                                    title="Capital Allocation", hole=0.4, color_discrete_sequence=px.colors.sequential.Plasma)
                    fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    if model_choice == "Hierarchical Risk Parity (HRP)":
                        # HRP Dendrogram (Simplified visualization of correlation)
                        st.markdown("##### 🔗 Cluster Structure (Correlation)")
                        st.plotly_chart(go.Figure(data=go.Heatmap(z=returns.corr().values, x=returns.columns, y=returns.columns, colorscale='Viridis')), use_container_width=True)

                    
            else:
                st.error("❌ Failed to download price data. Check ticker symbols.")
        else:
            st.info("💡 **Add at least 2 assets to enable portfolio optimization**")
            st.markdown("""
            ### Why Multiple Assets?
            - **Diversification** reduces risk without sacrificing returns
            - **Correlation** between assets determines optimal weights
            - **Markowitz Theory** finds the best risk-return balance
            
            **Try adding:**
            - Mix of stocks (AAPL, MSFT, GOOGL)
            - ETFs (SPY, QQQ)
            - Bonds (BTP=F, ZN=F)
            - Commodities (GC=F, CL=F)
            """)
    else:
        st.info("💡 **Add assets from the sidebar to start portfolio optimization**")
        st.markdown("""
        ### Modern Portfolio Theory
        Developed by Harry Markowitz (Nobel Prize 1990)
        
        **Key Concepts:**
        1. **Efficient Frontier** - Best possible portfolios for each risk level
        2. **Sharpe Ratio** - Measures return per unit of risk
        3. **Diversification** - Don't put all eggs in one basket
        4. **Correlation** - How assets move together
        
        Add 2+ assets to see the magic! ✨
        """)

# TAB 3: MACRO
with tab3:
    st.subheader("🌍 Global Markets Dashboard")
    
    # MARKET INDICES
    st.markdown("#### 📊 Real-Time Market Indices")
    macro_tickers = {
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC',
        'Dow Jones': '^DJI',
        'DAX': '^GDAXI',
        'Gold': 'GC=F',
        'Oil WTI': 'CL=F',
        'US 10Y': '^TNX',
        'VIX': '^VIX',
        'EUR/USD': 'EURUSD=X',
        'BTC': 'BTC-USD'
    }
    
    macro_data = get_prices(list(macro_tickers.values()), period="5d")
    
    if macro_data is not None:
        cols = st.columns(5)
        for idx, (name, ticker) in enumerate(macro_tickers.items()):
            if ticker in macro_data.columns:
                series = macro_data[ticker].dropna()
                if len(series) >= 2:
                    curr = series.iloc[-1]
                    prev = series.iloc[-2]
                    delta = ((curr - prev) / prev) * 100
                    
                    with cols[idx % 5]:
                        st.metric(name, f"{curr:,.2f}", f"{delta:+.2f}%",
                                 delta_color="normal" if "VIX" not in name else "inverse")
    
    st.markdown("---")
    
    # ECONOMIC DATA TABLE
    col_table, col_news = st.columns([1, 1])
    
    with col_table:
        st.markdown("#### 📈 Economic Snapshot")
        st.markdown("""
        <table style="width:100%; border-collapse: collapse; font-size:0.9rem;">
            <thead style="background: rgba(123, 47, 247, 0.2); color: #00f2ff;">
                <tr>
                    <th style="padding: 12px; text-align: left; border-bottom: 2px solid #00f2ff;">REGION</th>
                    <th style="padding: 12px; text-align: center; border-bottom: 2px solid #00f2ff;">GDP</th>
                    <th style="padding: 12px; text-align: center; border-bottom: 2px solid #00f2ff;">INFLATION</th>
                    <th style="padding: 12px; text-align: center; border-bottom: 2px solid #00f2ff;">RATE</th>
                </tr>
            </thead>
            <tbody style="color: #e8eaed;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <td style="padding: 10px;">🇺🇸 USA</td>
                    <td style="padding: 10px; text-align: center; color: #39d353;">+2.5%</td>
                    <td style="padding: 10px; text-align: center;">2.7%</td>
                    <td style="padding: 10px; text-align: center;">4.75%</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <td style="padding: 10px;">🇪🇺 Eurozone</td>
                    <td style="padding: 10px; text-align: center; color: #e3b341;">+0.7%</td>
                    <td style="padding: 10px; text-align: center;">2.4%</td>
                    <td style="padding: 10px; text-align: center;">3.00%</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <td style="padding: 10px;">🇨🇳 China</td>
                    <td style="padding: 10px; text-align: center; color: #39d353;">+4.8%</td>
                    <td style="padding: 10px; text-align: center; color: #39d353;">+0.3%</td>
                    <td style="padding: 10px; text-align: center;">3.45%</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <td style="padding: 10px;">🇬🇧 UK</td>
                    <td style="padding: 10px; text-align: center; color: #e3b341;">+0.5%</td>
                    <td style="padding: 10px; text-align: center; color: #39d353;">2.2%</td>
                    <td style="padding: 10px; text-align: center;">4.50%</td>
                </tr>
                <tr>
                    <td style="padding: 10px;">🇯🇵 Japan</td>
                    <td style="padding: 10px; text-align: center; color: #39d353;">+1.2%</td>
                    <td style="padding: 10px; text-align: center;">2.1%</td>
                    <td style="padding: 10px; text-align: center;">0.25%</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)
        st.caption(f"📅 Source: Central Banks & Gov Statistics (Updated: Feb 2026)")
    
    with col_news:
        st.markdown("#### 📰 Financial News")
        news = get_news()
        
        if news:
            for article in news[:6]:
                title = article.get('title', 'No title')
                link = article.get('link', '#')
                published = article.get('published', '')
                
                st.markdown(f"""
                <div style="background: rgba(26,31,58,0.5); padding: 12px; margin-bottom: 10px; 
                            border-left: 3px solid #7b2ff7; border-radius: 4px;">
                    <a href="{link}" target="_blank" style="color: #00f2ff; text-decoration: none; 
                       font-weight: 600; font-size: 0.9rem;">
                        {title[:80]}{'...' if len(title) > 80 else ''}
                    </a>
                    <div style="color: #8b949e; font-size: 0.75rem; margin-top: 4px;">
                        {published[:16] if published else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📡 News feed temporarily unavailable")
    
    st.markdown("---")

    # === COMPLEXITY SCIENCE SECTION ===
    st.subheader("🕸️ Complexity Science & Network Topology")
    st.caption("Advanced metrics from Chaos Theory, Network Science, and Econophysics")
    
    c_tab1, c_tab2, c_tab3, c_tab4 = st.tabs(["🌍 Trade Network & PageRank", "💰 Wealth Distribution (Pareto/Gini)", "🌀 Market Chaos & Entropy", "🛢️ Commodity DCCA"])
    
    # 1. NETWORK TOPOLOGY
    with c_tab1:
        try:
            st.markdown("#### Global Trade Network Criticality")
            st.info("💡 **PageRank** identifies 'Too Big To Fail' nodes in the global supply chain. Higher score = Higher systemic risk.")
            
            G, pagerank = generate_trade_network()
            
            # Plot Network
            pos = nx.spring_layout(G, seed=42)
            
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')

            node_x = []
            node_y = []
            node_text = []
            node_marker_size = []
            node_color = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                pr = pagerank[node]
                node_text.append(f"{node}: PageRank {pr:.3f}")
                node_marker_size.append(pr * 1000) # Size based on importance
                node_color.append(pr)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=[n for n in G.nodes()],
                textposition="top center",
                hoverinfo='text',
                hovertext=node_text,
                marker=dict(
                    showscale=True,
                    colorscale='Viridis',
                    size=node_marker_size,
                    color=node_color,
                    colorbar=dict(
                        thickness=15,
                        title='PageRank',
                        xanchor='left'
                    ),
                    line=dict(width=2)))
                    
            fig_net = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=dict(
                                text='Global Supply Chain Centrality (Simulation)',
                                font=dict(size=16)
                            ),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                            )
            st.plotly_chart(fig_net, use_container_width=True)
            
            # Sector Clustering
            st.markdown("#### 🔗 Sector Clustering Coefficients")
            st.caption("Measures how tightly interconnected assets are within a sector (Risk of Contagion)")
            
            clusters = get_sector_clusters()
            
            cluster_df = []
            for sector, assets in clusters.items():
                avg_corr = sum(assets.values()) / len(assets)
                cluster_df.append({"Sector": sector, "Clustering": avg_corr})
            
            cdf = pd.DataFrame(cluster_df)
            
            fig_cluster = px.bar(cdf, x="Sector", y="Clustering", color="Clustering", 
                                color_continuous_scale="Reds", title="Sector Risk Contagion (Interconnectivity)")
            fig_cluster.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,31,58,0.5)')
            st.plotly_chart(fig_cluster, use_container_width=True)
        except Exception as e:
            st.error(f"Error in Network Topology: {str(e)}")

    # 2. WEALTH DISTRIBUTION
    with c_tab2:
        try:
            st.markdown("#### ⚖️ Wealth Distribution Physics (Pareto & Gini)")
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("##### Global Wealth Gini Index")
                # Simulated data for visualization
                gini_data = pd.DataFrame({
                    'Region': ['Africa', 'LatAm', 'Asia', 'N. America', 'Europe'],
                    'Gini': [0.58, 0.52, 0.45, 0.42, 0.32]
                })
                
                fig_gini = px.bar(gini_data, x='Region', y='Gini', color='Gini', 
                                 color_continuous_scale='Magma', title="Regional Inequality (High Gini = High Instability)")
                fig_gini.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,31,58,0.5)')
                st.plotly_chart(fig_gini, use_container_width=True)
                
            with c2:
                st.markdown("##### Pareto Wealth Distribution (Power Law)")
                # Power law simulation
                x = np.linspace(1, 10, 100)
                y = 1 / (x**1.16) # Alpha approx 1.16 for wealth
                
                fig_pareto = go.Figure()
                fig_pareto.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', name='Wealth Distribution'))
                fig_pareto.update_layout(title="The 80/20 Rule Visualized", 
                                       xaxis_title="Wealth ($ Trillions)", yaxis_title="Population (Billions)",
                                       paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,31,58,0.5)')
                st.plotly_chart(fig_pareto, use_container_width=True)
                
            st.caption("📚 **Data Sources:** World Bank (Gini Index), Credit Suisse Global Wealth Report (Wealth Distribution). Metrics are illustrative.")
        except Exception as e:
            st.error(f"Error in Wealth Distribution: {str(e)}")

    # 3. CHAOS & ENTROPY
    with c_tab3:
        try:
            st.markdown("#### 🌀 Market Chaos & Multifractality")
            
            # Fetch long-term data for Chaos Analysis (needs at least 1 year)
            chaos_data = get_prices(list(macro_tickers.values()), period="1y")

            if chaos_data is not None and not chaos_data.empty:
                # Calculate Hurst and Entropy for major indices
                chaos_metrics = []
                
                for name, ticker in macro_tickers.items():
                    if ticker in chaos_data.columns and "EUR" not in name and "10Y" not in name:
                        series = chaos_data[ticker].dropna()
                        # Need at least 100 points for reliable Hurst
                        if len(series) > 100:
                            h = calculate_hurst(series)
                            e = calculate_shannon_entropy(series)
                            
                            regime = "Random Walk"
                            if h > 0.6: regime = "Persistent Trend 🚀"
                            elif h < 0.4: regime = "Mean Reverting 🔄"
                            
                            chaos_metrics.append({
                                "Asset": name,
                                "Hurst Exp (H)": h,
                                "Regime": regime,
                                "Entropy (S)": e,
                                "Last Price": f"{series.iloc[-1]:.2f} {('USD' if 'BTC' in name or 'Gold' in name or 'Oil' in name else '')}"
                            })
                
                if chaos_metrics:
                    chaos_df = pd.DataFrame(chaos_metrics)
                    
                    # Format for display
                    st.dataframe(
                        chaos_df.style.format({
                            "Hurst Exp (H)": "{:.3f}",
                            "Entropy (S)": "{:.3f}"
                        }).background_gradient(subset=['Hurst Exp (H)'], cmap='coolwarm'),
                        use_container_width=True, 
                        hide_index=True
                    )
                    
                    st.info("""
                    **Legend:**
                    - **H > 0.5 (Red/Warm):** Trend persistente (il mercato "ricorda" la direzione).
                    - **H < 0.5 (Blue/Cool):** Mean Reverting (il mercato tende a invertire).
                    - **High Entropy:** Alta incertezza/sorpresa. (Valori più alti = più chaos)
                    """)
                else:
                     st.warning("Insufficient data points for meaningful Chaos calculation.")
            else:
                 st.warning("Could not fetch historical data for Chaos analysis.")
        except Exception as e:
            st.error(f"Error in Chaos & Entropy: {str(e)}")

    # 4. COMMODITY DCCA
    with c_tab4:
        try:
            st.markdown("#### 🛢️ Commodity Dynamic Cross-Correlation (DCCA)")
            st.caption("Analyzing non-stationary correlations between Commodities, Gold, and Equities over time.")
            
            # Fetch specifically for DCCA
            dcca_tickers = ['GC=F', 'CL=F', '^GSPC']
            dcca_data = get_prices(dcca_tickers, period="2y")
            
            if dcca_data is not None and not dcca_data.empty:
                # Align data
                # Check column names (yfinance structure varies)
                cols = dcca_data.columns
                
                # Robust ticker selection
                gold_col = next((c for c in cols if 'GC=F' in c), 'GC=F')
                oil_col = next((c for c in cols if 'CL=F' in c), 'CL=F')
                spx_col = next((c for c in cols if '^GSPC' in c), '^GSPC')
                
                if gold_col in dcca_data.columns and oil_col in dcca_data.columns and spx_col in dcca_data.columns:
                    gold = dcca_data[gold_col].dropna()
                    oil = dcca_data[oil_col].dropna()
                    spx = dcca_data[spx_col].dropna()
                    
                    # Calculate DCCA
                    rho_gold_oil = calculate_dcca(gold, oil, window=60)
                    rho_gold_spx = calculate_dcca(gold, spx, window=60)
                    
                    # Plot
                    fig_dcca = go.Figure()
                    fig_dcca.add_trace(go.Scatter(x=rho_gold_oil.index, y=rho_gold_oil, name='Gold vs Oil', line=dict(color='#ffaa00')))
                    fig_dcca.add_trace(go.Scatter(x=rho_gold_spx.index, y=rho_gold_spx, name='Gold vs S&P 500', line=dict(color='#00f2ff')))
                    
                    fig_dcca.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
                    fig_dcca.update_layout(title="Dynamic Correlation Evolution (Rolling DCCA Proxy)", 
                                         yaxis_title="Correlation coefficient (ρ)",
                                         hovermode="x unified",
                                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,31,58,0.5)')
                    st.plotly_chart(fig_dcca, use_container_width=True)
                    
                    st.info("ℹ️ **Interpretation:** Positive values = Move together. Negative values = Hedge/Safe Haven behavior.")
                else:
                    st.warning("Could not find required tickers in data.")
            else:
                 st.warning("Insufficient data for DCCA analysis.")
        except Exception as e:
            st.error(f"Error in DCCA: {str(e)}")
    
    st.markdown("---")
    
    # CENTRAL BANK CALENDAR
    st.markdown("#### 🏦 Upcoming Central Bank Meetings")
    cb_cols = st.columns(4)
    
    cb_cols[0].markdown("""
    **🇺🇸 FED**  
    Next: Mar 19, 2026  
    Expected: Hold 4.75%
    """)
    
    cb_cols[1].markdown("""
    **🇪🇺 ECB**  
    Next: Mar 6, 2026  
    Expected: Cut -0.25%
    """)
    
    cb_cols[2].markdown("""
    **🇬🇧 BoE**  
    Next: Mar 20, 2026  
    Expected: Hold 4.50%
    """)
    
    cb_cols[3].markdown("""
    **🇯🇵 BoJ**  
    Next: Mar 18, 2026  
    Expected: Hold 0.25%
    """)


# TAB 4: ALERTS
with tab4:
    st.subheader("🔔 Price Alert System")
    
    with st.form("alert_form"):
        # Select box needs options. If portfolio is empty, provide a default or empty list but handle gracefully
        port_tickers = st.session_state.portfolio['Ticker'].unique() if not st.session_state.portfolio.empty else []
        al_ticker = st.selectbox("Asset:", port_tickers) if len(port_tickers) > 0 else st.text_input("Asset (Manual):")
        
        al_price = st.number_input("Target Price:", min_value=0.01, value=100.0)
        al_type = st.radio("Type:", ["Above", "Below"])
        
        if st.form_submit_button("Create Alert"):
            if al_ticker:
                st.session_state.alerts.append({
                    'ticker': al_ticker,
                    'price': al_price,
                    'type': al_type,
                    'created': datetime.now().isoformat()
                })
                
                # SAVE TO FIRESTORE
                auth_handler = auth.FirebaseAuth()
                auth_handler.save_user_alerts(st.session_state.user_id, st.session_state.alerts)
                
                st.success(f"✅ Alert created: {al_ticker} {al_type.lower()} ${al_price}")
                st.rerun()
            else:
                st.warning("Please select an asset first")
    
    if st.session_state.alerts:
        st.markdown("#### Active Alerts")
        for alert in st.session_state.alerts:
            st.markdown(f"""
            <div class="alert-box">
                <strong>{str(alert['ticker'])}</strong> - Notify when price goes {str(alert['type']).lower()} ${float(alert['price']):.2f}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # INTELLIGENT ALERTS SECTIONS
    st.markdown("### 🧠 Quantum Intelligence Center")
    
    monitoring_list = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', # Big Tech
        'BTC-USD', 'ETH-USD', 'SOL-USD', # Crypto
        'GC=F', 'CL=F', # Commodities
        'JPM', 'V', 'LLY', 'XOM' # Finance/Pharma/Energy
    ]
    
    if st.button("🔄 Scan Market for Quantum Opportunities"):
        with st.spinner("🔍 Scanning market dimensions... (Physics + Fundamentals + Technicals)"):
            scan_data = get_prices(monitoring_list, period="6mo")
            
            if scan_data is not None:
                opportunities = []
                movers = []
                
                for ticker in monitoring_list:
                    try:
                        # Handle Data Structure
                        if isinstance(scan_data, pd.DataFrame) and isinstance(scan_data.columns, pd.MultiIndex):
                             # Multi-level columns if multiple tickers
                             if ticker in scan_data.columns.get_level_values(0): # Check level 0
                                 info_col = [c for c in scan_data.columns if c[0] == ticker or c == ticker] 
                                 # This is getting messy with yfinance formats.
                                 # Simpler: extraction
                                 series = scan_data[ticker] if ticker in scan_data.columns else None
                        else:
                             series = scan_data[ticker] if ticker in scan_data.columns else None
                        
                        if series is None:
                             # Try looking for it if flattened
                             # yfinance 'Adj Close' usually
                             pass

                        # Fallback: Re-fetch if bulk failed structure
                        if series is None or series.empty:
                            series = get_prices(ticker, period="6mo")
                        
                        if series is None or len(series) < 50:
                            continue
                            
                        # Calc Metrics
                        curr_price = series.iloc[-1]
                        day_change = ((curr_price - series.iloc[-2]) / series.iloc[-2]) * 100
                        
                        rsi = calc_rsi(series).iloc[-1]
                        hurst = calculate_hurst(series)
                        
                        # Logic for Movers
                        movers.append({
                            'Ticker': ticker,
                            'Price': curr_price,
                            'Change %': day_change
                        })
                        
                        # Logic for Picks (Quantum Setup)
                        # Setup 1: Strong Trend (H > 0.6) + Momentum (RSI > 50 but < 70) -> MOMENTUM BUY
                        # Setup 2: Mean Reversion (H < 0.4) + Oversold (RSI < 30) -> REVERSAL BUY
                        
                        signal_type = None
                        confidence = "Medium"
                        
                        if hurst > 0.6 and 50 < rsi < 70 and day_change > 0:
                            signal_type = "🚀 MOMENTUM BREAKOUT"
                            confidence = "High" if hurst > 0.7 else "Medium"
                        elif hurst < 0.4 and rsi < 35:
                            signal_type = "💎 OVERSOLD REVERSAL"
                            confidence = "High" if rsi < 25 else "Medium"
                        elif rsi > 75:
                             signal_type = "⚠️ OVERBOUGHT (RISK)"
                             confidence = "High"
                        
                        if signal_type:
                            opportunities.append({
                                'Ticker': ticker,
                                'Signal': signal_type,
                                'Confidence': confidence,
                                'Price': f"${curr_price:.2f}",
                                'Metrics': f"RSI: {rsi:.0f} | H: {hurst:.2f}"
                            })
                            
                    except Exception as e:
                        continue
                
                # DISPLAY MOVERS
                st.markdown("#### 🌊 Market Movers (Real-Time)")
                movers_df = pd.DataFrame(movers).sort_values('Change %', ascending=False)
                
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown("##### 🟢 Top Gainers")
                    st.dataframe(movers_df.head(5).style.format({'Price': "${:.2f}", 'Change %': "{:+.2f}%"}), hide_index=True)
                with m2:
                    st.markdown("##### 🔴 Top Losers")
                    st.dataframe(movers_df.tail(5).sort_values('Change %').style.format({'Price': "${:.2f}", 'Change %': "{:+.2f}%"}), hide_index=True)
                
                st.markdown("---")
                
                # DISPLAY PICKS
                st.markdown("#### ⚡ Quantum Picks")
                if opportunities:
                    for opp in opportunities:
                        color = "#00ff00" if "BUY" in opp['Signal'] or "MOMENTUM" in opp['Signal'] or "REVERSAL" in opp['Signal'] else "#ff0000"
                        st.markdown(f"""
                        <div style="border: 1px solid {color}; border-radius: 8px; padding: 15px; margin-bottom: 10px; background: rgba(0,0,0,0.2);">
                            <div style="display: flex; justify-content: space-between; align_items: center;">
                                <h3 style="margin: 0; color: {color};">{opp['Signal']}</h3>
                                <span style="font-weight: bold; font-size: 1.2em;">{opp['Ticker']}</span>
                            </div>
                            <p style="margin: 5px 0;">Price: <strong>{opp['Price']}</strong> | Confidence: <strong>{opp['Confidence']}</strong></p>
                            <span style="font-size: 0.9em; color: #888;">{opp['Metrics']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No strong quantum setups detected in the monitored list right now.")
                    
            else:
                st.error("Could not fetch market data.")
    else:
        st.info("👋 Click 'Scan Market' to generate real-time Quantum signals and ranks.")

# TAB 5: AI ASSISTANT
with tab5:
    st.subheader("🤖 Quantum AI Assistant")
    st.markdown("*Il tuo tutor finanziario personale - Fai qualsiasi domanda su grafici, metriche e concetti finanziari*")
    
    # Quick Actions
    col_quick1, col_quick2, col_quick3 = st.columns(3)
    
    with col_quick1:
        if st.button("📊 Tips sui Grafici", use_container_width=True):
            response = get_chart_tips()
            st.session_state.chat_history.append({"question": "Tips sui grafici", "answer": response})
    
    with col_quick2:
        if st.button("💼 Analisi Fondamentale", use_container_width=True):
            response = get_fundamental_tips()
            st.session_state.chat_history.append({"question": "Analisi fondamentale", "answer": response})
    
    with col_quick3:
        if st.button("🔄 Cancella Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    st.markdown("---")
    
    # Chat Interface
    with st.form("ai_chat_form", clear_on_submit=True):
        user_question = st.text_input("💬 Fai una domanda:", 
                                      placeholder="Es: Cosa significa RSI? Come si legge la frontiera efficiente?")
        ask_btn = st.form_submit_button("🚀 CHIEDI")
        
        if ask_btn and user_question:
            response = get_ai_response(user_question)
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": response
            })
            st.rerun()
    
    # Display Chat History (reverse order - newest first)
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversazione")
        for idx, chat in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"""
            <div class="chat-bubble">
                <strong>❓ Tu:</strong> {chat['question']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(chat['answer'])
            st.markdown("---")
    else:
        # Welcome message with suggestions
        st.info("👋 **Benvenuto! Sono il tuo assistente AI.**")
        st.markdown("""
        ### 💡 Domande Frequenti:
        
        **📊 Indicatori Tecnici:**
        - "Cos'è l'RSI e come funziona?"
        - "Cosa sono le medie mobili SMA?"
        - "Come si interpreta la volatilità?"
        
        **💰 Metriche Fondamentali:**
        - "Spiega il P/E ratio"
        - "Cos'è la Market Cap?"
        - "Come funziona il Dividend Yield?"
        
        **🎯 Portfolio & Grafici:**
        - "Come si legge la frontiera efficiente?"
        - "Cosa significa correlazione tra asset?"
        - "Spiega lo Sharpe Ratio"
        
        **🔍 Interpretazione:**
        - "Come leggere i grafici dell'app?"
        - "Cosa significano i segnali BUY/SELL?"
        - "Come usare il portfolio optimizer?"
        
        ### 🎓 Esempi Pratici Inclusi
        Ogni risposta include esempi reali e interpretazioni pratiche!
        
        ---
        **Prova a chiedere qualcosa!** 👆
        """)
