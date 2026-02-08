# Complete Asset Database with Full Names
# Format: (Ticker, Full Name, Category)

ASSET_DATABASE = [
    # === US STOCKS - MEGA CAP ===
    ("AAPL", "Apple Inc.", "US Stocks"),
    ("MSFT", "Microsoft Corporation", "US Stocks"),
    ("GOOGL", "Alphabet Inc. (Google)", "US Stocks"),
    ("AMZN", "Amazon.com Inc.", "US Stocks"),
    ("NVDA", "NVIDIA Corporation", "US Stocks"),
    ("META", "Meta Platforms (Facebook)", "US Stocks"),
    ("TSLA", "Tesla Inc.", "US Stocks"),
    ("BRK-B", "Berkshire Hathaway", "US Stocks"),
    ("V", "Visa Inc.", "US Stocks"),
    ("JPM", "JPMorgan Chase", "US Stocks"),
    ("WMT", "Walmart Inc.", "US Stocks"),
    ("MA", "Mastercard Inc.", "US Stocks"),
    ("PG", "Procter & Gamble", "US Stocks"),
    ("JNJ", "Johnson & Johnson", "US Stocks"),
    ("UNH", "UnitedHealth Group", "US Stocks"),
    ("HD", "Home Depot", "US Stocks"),
    ("DIS", "Walt Disney Company", "US Stocks"),
    ("NFLX", "Netflix Inc.", "US Stocks"),
    ("PYPL", "PayPal Holdings", "US Stocks"),
    ("ADBE", "Adobe Inc.", "US Stocks"),
    ("CSCO", "Cisco Systems", "US Stocks"),
    ("INTC", "Intel Corporation", "US Stocks"),
    ("AMD", "Advanced Micro Devices", "US Stocks"),
    ("CRM", "Salesforce Inc.", "US Stocks"),
    ("NKE", "Nike Inc.", "US Stocks"),
    ("PFE", "Pfizer Inc.", "US Stocks"),
    ("KO", "Coca-Cola Company", "US Stocks"),
    ("PEP", "PepsiCo Inc.", "US Stocks"),
    ("ABBV", "AbbVie Inc.", "US Stocks"),
    ("MRK", "Merck & Co.", "US Stocks"),
    ("BAC", "Bank of America", "US Stocks"),
    ("WFC", "Wells Fargo", "US Stocks"),
    ("T", "AT&T Inc.", "US Stocks"),
    ("VZ", "Verizon Communications", "US Stocks"),
    ("AVGO", "Broadcom Inc.", "US Stocks"),
    ("QCOM", "Qualcomm Inc.", "US Stocks"),
    ("TXN", "Texas Instruments", "US Stocks"),
    ("ORCL", "Oracle Corporation", "US Stocks"),
    ("IBM", "IBM Corporation", "US Stocks"),
    ("UBER", "Uber Technologies", "US Stocks"),
    ("ABNB", "Airbnb Inc.", "US Stocks"),
    ("SQ", "Block Inc. (Square)", "US Stocks"),
    ("SHOP", "Shopify Inc.", "US Stocks"),
    ("COIN", "Coinbase Global", "US Stocks"),
    ("SNOW", "Snowflake Inc.", "US Stocks"),
    ("ZM", "Zoom Video Communications", "US Stocks"),
    ("PLTR", "Palantir Technologies", "US Stocks"),
    
    # === EUROPEAN STOCKS ===
    ("ENEL.MI", "Enel S.p.A. (Italy)", "European Stocks"),
    ("ISP.MI", "Intesa Sanpaolo (Italy)", "European Stocks"),
    ("UCG.MI", "UniCredit (Italy)", "European Stocks"),
    ("ENI.MI", "Eni S.p.A. (Italy)", "European Stocks"),
    ("TIT.MI", "Telecom Italia", "European Stocks"),
    ("STLA.MI", "Stellantis (Fiat Chrysler)", "European Stocks"),
    ("SAN.MC", "Banco Santander (Spain)", "European Stocks"),
    ("TEF.MC", "Telefónica (Spain)", "European Stocks"),
    ("IBE.MC", "Iberdrola (Spain)", "European Stocks"),
    ("SAP", "SAP SE (Germany)", "European Stocks"),
    ("SIE.DE", "Siemens AG (Germany)", "European Stocks"),
    ("VOW3.DE", "Volkswagen AG", "European Stocks"),
    ("BMW.DE", "BMW AG", "European Stocks"),
    ("OR.PA", "L'Oréal (France)", "European Stocks"),
    ("MC.PA", "LVMH (France)", "European Stocks"),
    ("SAN.PA", "Sanofi (France)", "European Stocks"),
    ("AI.PA", "Air Liquide (France)", "European Stocks"),
    ("ASML", "ASML Holding (Netherlands)", "European Stocks"),
    ("NESN.SW", "Nestlé (Switzerland)", "European Stocks"),
    ("NOVN.SW", "Novartis (Switzerland)", "European Stocks"),
    ("ROG.SW", "Roche Holding (Switzerland)", "European Stocks"),
    
    # === ETFs - US ===
    ("SPY", "SPDR S&P 500 ETF", "ETFs"),
    ("QQQ", "Invesco QQQ (Nasdaq 100)", "ETFs"),
    ("VOO", "Vanguard S&P 500 ETF", "ETFs"),
    ("VTI", "Vanguard Total Stock Market", "ETFs"),
    ("IVV", "iShares Core S&P 500", "ETFs"),
    ("DIA", "SPDR Dow Jones Industrial", "ETFs"),
    ("IWM", "iShares Russell 2000", "ETFs"),
    ("VEA", "Vanguard FTSE Developed Markets", "ETFs"),
    ("VWO", "Vanguard FTSE Emerging Markets", "ETFs"),
    ("AGG", "iShares Core US Aggregate Bond", "ETFs"),
    ("BND", "Vanguard Total Bond Market", "ETFs"),
    ("GLD", "SPDR Gold Shares", "ETFs"),
    ("SLV", "iShares Silver Trust", "ETFs"),
    ("USO", "United States Oil Fund", "ETFs"),
    ("XLE", "Energy Select Sector SPDR", "ETFs"),
    ("XLF", "Financial Select Sector SPDR", "ETFs"),
    ("XLK", "Technology Select Sector SPDR", "ETFs"),
    ("XLV", "Health Care Select Sector SPDR", "ETFs"),
    ("XLP", "Consumer Staples Select SPDR", "ETFs"),
    ("VNQ", "Vanguard Real Estate ETF", "ETFs"),
    ("TLT", "iShares 20+ Year Treasury Bond", "ETFs"),
    ("HYG", "iShares iBoxx High Yield Corporate", "ETFs"),
    ("EEM", "iShares MSCI Emerging Markets", "ETFs"),
    ("EFA", "iShares MSCI EAFE", "ETFs"),
    ("IEMG", "iShares Core MSCI Emerging Markets", "ETFs"),
    
    # === ETFs - EUROPEAN ===
    ("SWDA.MI", "iShares Core MSCI World", "ETFs Europe"),
    ("VWCE.DE", "Vanguard FTSE All-World", "ETFs Europe"),
    ("EIMI.MI", "iShares Core MSCI Emerging Markets IMI", "ETFs Europe"),
    ("CSPX.MI", "iShares Core S&P 500", "ETFs Europe"),
    ("VUAA.MI", "Vanguard S&P 500", "ETFs Europe"),
    ("VUSA.L", "Vanguard S&P 500 (London)", "ETFs Europe"),
    
    # === CRYPTOCURRENCIES ===
    ("BTC-USD", "Bitcoin", "Crypto"),
    ("ETH-USD", "Ethereum", "Crypto"),
    ("BNB-USD", "Binance Coin", "Crypto"),
    ("XRP-USD", "Ripple XRP", "Crypto"),
    ("ADA-USD", "Cardano", "Crypto"),
    ("SOL-USD", "Solana", "Crypto"),
    ("DOGE-USD", "Dogecoin", "Crypto"),
    ("DOT-USD", "Polkadot", "Crypto"),
    ("MATIC-USD", "Polygon", "Crypto"),
    ("AVAX-USD", "Avalanche", "Crypto"),
    ("LINK-USD", "Chainlink", "Crypto"),
    ("UNI-USD", "Uniswap", "Crypto"),
    ("LTC-USD", "Litecoin", "Crypto"),
    ("BCH-USD", "Bitcoin Cash", "Crypto"),
    ("ATOM-USD", "Cosmos", "Crypto"),
    
    # === COMMODITIES & FUTURES ===
    ("GC=F", "Gold Futures", "Commodities"),
    ("SI=F", "Silver Futures", "Commodities"),
    ("CL=F", "Crude Oil WTI Futures", "Commodities"),
    ("NG=F", "Natural Gas Futures", "Commodities"),
    ("HG=F", "Copper Futures", "Commodities"),
    ("PL=F", "Platinum Futures", "Commodities"),
    ("PA=F", "Palladium Futures", "Commodities"),
    ("ZC=F", "Corn Futures", "Commodities"),
    ("ZW=F", "Wheat Futures", "Commodities"),
    ("ZS=F", "Soybean Futures", "Commodities"),
    
    # === BONDS & TREASURY FUTURES ===
    ("BTP=F", "BTP Italy 10Y Futures", "Bonds"),
    ("FGBL=F", "Euro-Bund German 10Y Futures", "Bonds"),
    ("ZN=F", "US Treasury Note 10Y Futures", "Bonds"),
    ("ZB=F", "US Treasury Bond 30Y Futures", "Bonds"),
    ("ZF=F", "US Treasury Note 5Y Futures", "Bonds"),
    ("ZT=F", "US Treasury Note 2Y Futures", "Bonds"),
    
    # === INDICES ===
    ("^GSPC", "S&P 500 Index", "Indices"),
    ("^DJI", "Dow Jones Industrial Average", "Indices"),
    ("^IXIC", "NASDAQ Composite", "Indices"),
    ("^RUT", "Russell 2000", "Indices"),
    ("^VIX", "CBOE Volatility Index (VIX)", "Indices"),
    ("^FTSE", "FTSE 100 (UK)", "Indices"),
    ("^GDAXI", "DAX (Germany)", "Indices"),
    ("^FCHI", "CAC 40 (France)", "Indices"),
    ("^N225", "Nikkei 225 (Japan)", "Indices"),
    ("^HSI", "Hang Seng (Hong Kong)", "Indices"),
    ("000001.SS", "Shanghai Composite", "Indices"),
    ("^STOXX50E", "Euro Stoxx 50", "Indices"),
    ("FTSEMIB.MI", "FTSE MIB (Italy)", "Indices"),
    ("^IBEX", "IBEX 35 (Spain)", "Indices"),
    ("^TNX", "US 10-Year Treasury Yield", "Indices"),
    
    # === FOREX ===
    ("EURUSD=X", "EUR/USD", "Forex"),
    ("GBPUSD=X", "GBP/USD", "Forex"),
    ("USDJPY=X", "USD/JPY", "Forex"),
    ("AUDUSD=X", "AUD/USD", "Forex"),
    ("USDCAD=X", "USD/CAD", "Forex"),
    ("USDCHF=X", "USD/CHF", "Forex"),
    ("EURGBP=X", "EUR/GBP", "Forex"),
    ("EURJPY=X", "EUR/JPY", "Forex"),
]

def search_assets(query):
    """Search assets by ticker or name"""
    query = query.upper().strip()
    results = []
    
    for ticker, name, category in ASSET_DATABASE:
        # Match ticker or name
        if query in ticker.upper() or query in name.upper():
            results.append({
                'ticker': ticker,
                'name': name,
                'category': category,
                'display': f"{ticker} - {name}"
            })
    
    return results

def get_all_tickers():
    """Get list of all tickers"""
    return [item[0] for item in ASSET_DATABASE]

def get_ticker_info(ticker):
    """Get full info for a ticker"""
    for t, name, category in ASSET_DATABASE:
        if t == ticker:
            return {'ticker': t, 'name': name, 'category': category}
    return None

def get_categories():
    """Get unique categories"""
    return sorted(list(set([item[2] for item in ASSET_DATABASE])))

def get_assets_by_category(category):
    """Get all assets in a category"""
    return [(t, n) for t, n, c in ASSET_DATABASE if c == category]
