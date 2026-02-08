# AI Assistant for Financial Education
# Knowledge base for explaining charts, metrics, and financial concepts

AI_KNOWLEDGE_BASE = {
    # TECHNICAL INDICATORS
    "rsi": {
        "name": "RSI (Relative Strength Index)",
        "explanation": """
**RSI** misura la forza del movimento dei prezzi su una scala 0-100.

📊 **Come leggerlo:**
- **RSI < 30** = IPERVENDUTO → Il prezzo potrebbe essere troppo basso, possibile rimbalzo
- **RSI > 70** = IPERCOMPRATO → Il prezzo potrebbe essere troppo alto, possibile correzione
- **RSI 40-60** = NEUTRALE → Nessun segnale forte

💡 **Esempio pratico:**
Se un'azione costa €100 e l'RSI è 25, significa che è stata venduta troppo e potrebbe risalire presto.
        """,
        "keywords": ["rsi", "relative strength", "ipervenduto", "ipercomprato", "oversold", "overbought"]
    },
    
    "sma": {
        "name": "SMA (Simple Moving Average)",
        "explanation": """
**SMA** è la media del prezzo degli ultimi N giorni. Indica il trend.

📈 **Tipologie:**
- **SMA 20** = Media degli ultimi 20 giorni (trend breve)
- **SMA 50** = Media degli ultimi 50 giorni (trend medio)
- **SMA 200** = Media degli ultimi 200 giorni (trend lungo)

🔍 **Come usarla:**
- Prezzo > SMA = TREND POSITIVO (rialzista)
- Prezzo < SMA = TREND NEGATIVO (ribassista)
- Quando SMA50 incrocia SMA200 dal basso = "Golden Cross" (segnale BUY)

💡 **Esempio:**
Se Apple costa $180 e SMA200 è $160, il trend è positivo.
        """,
        "keywords": ["sma", "media mobile", "moving average", "trend", "golden cross"]
    },
    
    "sharpe": {
        "name": "Sharpe Ratio",
        "explanation": """
**Sharpe Ratio** misura quanto rendimento ottieni per ogni unità di rischio.

📐 **Formula:** (Rendimento - Tasso Risk-Free) / Volatilità

🎯 **Interpretazione:**
- **Sharpe > 1** = BUONO (il rischio è ben compensato)
- **Sharpe > 2** = OTTIMO (rendimento eccellente per il rischio)
- **Sharpe < 1** = SCARSO (troppo rischio per poco guadagno)

💡 **Esempio:**
Portfolio A: +15% return, Sharpe 1.8
Portfolio B: +20% return, Sharpe 0.9
→ A è migliore! Anche se rende meno, il rischio è molto più basso.
        """,
        "keywords": ["sharpe", "ratio", "rischio", "rendimento", "risk adjusted"]
    },
    
    "pe": {
        "name": "P/E Ratio (Price-to-Earnings)",
        "explanation": """
**P/E Ratio** indica quante volte l'utile annuale paghi comprando l'azione.

📐 **Formula:** Prezzo Azione / Utile per Azione

💰 **Interpretazione:**
- **P/E 10-15** = SOTTOVALUTATO (mercato non crede nella crescita)
- **P/E 15-25** = NORMALE (valutazione equilibrata)
- **P/E > 30** = SOPRAVVALUTATO (mercato si aspetta forte crescita)

💡 **Esempio:**
Tesla P/E = 60 → Paghi 60 anni di utili
Apple P/E = 28 → Paghi 28 anni di utili
→ Tesla è più costosa, ma il mercato si aspetta crescita maggiore

⚠️ **Attenzione:** Tech e startup hanno P/E alti perché crescono velocemente.
        """,
        "keywords": ["pe", "p/e", "price earnings", "valutazione", "utile", "earnings"]
    },
    
    "marketcap": {
        "name": "Market Cap (Capitalizzazione di Mercato)",
        "explanation": """
**Market Cap** è il valore totale di un'azienda in borsa.

📐 **Formula:** Prezzo Azione × Numero Azioni

📊 **Categorie:**
- **Large Cap** (> $10B) = Grandi aziende stabili (Apple, Microsoft)
- **Mid Cap** ($2-10B) = Aziende medie in crescita
- **Small Cap** (< $2B) = Piccole aziende, più rischiose

💡 **Esempio:**
Apple: 3 miliardi di azioni × $180 = **$540 miliardi Market Cap**

🎯 **Perché è importante:**
Large Cap = più stabili, meno volatili
Small Cap = più rischiose, potenziale guadagno maggiore
        """,
        "keywords": ["market cap", "capitalizzazione", "mcap", "valore azienda"]
    },
    
    "dividend": {
        "name": "Dividend Yield (Rendimento da Dividendi)",
        "explanation": """
**Dividend Yield** mostra quanto guadagni in dividendi annuali.

📐 **Formula:** (Dividendo Annuale / Prezzo Azione) × 100

💰 **Interpretazione:**
- **Yield 0-2%** = Basso (aziende growth che reinvestono tutto)
- **Yield 2-4%** = Buono (equilibrio tra crescita e dividendi)
- **Yield > 6%** = Alto (attenzione: potrebbe essere insostenibile)

💡 **Esempio:**
Compri azioni Enel a €6, dividendo annuale €0.36
Yield = (0.36 / 6) × 100 = **6%**
→ Ogni anno ricevi il 6% del tuo investimento in dividendi

🎯 **Quando preferirlo:**
- Investitori che vogliono reddito passivo
- Pensionati che vivono di rendita
- Mercati laterali (senza crescita)
        """,
        "keywords": ["dividend", "dividendo", "yield", "rendimento", "cedola"]
    },
    
    "correlation": {
        "name": "Correlation Matrix (Matrice di Correlazione)",
        "explanation": """
**Correlation** misura come si muovono insieme due asset (-1 a +1).

🔢 **Valori:**
- **+1** = Perfetta correlazione positiva (si muovono insieme)
- **0** = Nessuna correlazione (movimenti indipendenti)
- **-1** = Perfetta correlazione negativa (si muovono opposti)

🎨 **Colori nella Heatmap:**
- 🟢 **VERDE** = Correlazione negativa (ottimo per diversificazione!)
- 🟡 **GIALLO** = Correlazione neutra
- 🔴 **ROSSO** = Correlazione positiva (rischio concentrato)

💡 **Esempio portfolio:**
```
        AAPL    GLD
AAPL    1.00   -0.15
GLD    -0.15    1.00
```
→ Apple e Gold hanno correlazione negativa (-0.15)
→ Quando Apple scende, Gold tende a salire = DIVERSIFICAZIONE PERFETTA!

🎯 **Regola d'oro:** Cerca asset con correlazione < 0.3 per ridurre il rischio.
        """,
        "keywords": ["correlation", "correlazione", "heatmap", "diversificazione", "asset"]
    },
    
    "efficient_frontier": {
        "name": "Efficient Frontier (Frontiera Efficiente)",
        "explanation": """
**Efficient Frontier** mostra tutti i portfolio ottimali per ogni livello di rischio.

📊 **Grafico:**
- Asse X = RISCHIO (Volatilità)
- Asse Y = RENDIMENTO
- Ogni punto = un portfolio con pesi diversi

🎯 **Come leggerlo:**
- Punti in ALTO A SINISTRA = I MIGLIORI (massimo return, minimo rischio)
- Punti in BASSO A DESTRA = I PEGGIORI (basso return, alto rischio)
- ⭐ **STELLA** = Portfolio OTTIMALE (massimo Sharpe Ratio)

💡 **Esempio:**
Il grafico simula 5000 portfolio random. La stella ti dice:
"Con questi asset, la combinazione migliore è 40% AAPL, 30% SPY, 30% Gold"

🏆 **Obiettivo:** Stare sulla "frontiera" = massimo rendimento per quel rischio.
        """,
        "keywords": ["efficient frontier", "frontiera efficiente", "markowitz", "portfolio", "ottimizzazione"]
    },
    
    "volatility": {
        "name": "Volatility (Volatilità)",
        "explanation": """
**Volatilità** misura quanto variano i prezzi. Alto = più rischio.

📊 **Livelli:**
- **< 15%** = Bassa volatilità (asset stabili: bonds, utilities)
- **15-30%** = Media volatilità (S&P 500, blue chips)
- **> 50%** = Alta volatilità (crypto, small cap, tech growth)

💡 **Esempi:**
- **Titoli di Stato** = Volatilità 3-5% (molto stabili)
- **Apple, Microsoft** = Volatilità 20-30% (medie)
- **Bitcoin** = Volatilità 70-100% (altissima)

🎯 **Nel Portfolio Optimizer:**
Il grafico mostra volatilità annuale. Se vedi 20%, significa che il tuo portfolio potrebbe salire/scendere del 20% in un anno.

⚠️ **Regola:** Più alta la volatilità, più alto deve essere il rendimento atteso.
        """,
        "keywords": ["volatility", "volatilità", "rischio", "variazione", "deviazione standard"]
    },
    
    "eps": {
        "name": "EPS (Earnings Per Share)",
        "explanation": """
**EPS** è l'utile netto diviso per il numero di azioni.

📐 **Formula:** Utile Netto / Numero Azioni

💰 **Interpretazione:**
- EPS in CRESCITA = Azienda sta guadagnando di più (POSITIVO)
- EPS in CALO = Azienda sta guadagnando di meno (NEGATIVO)
- EPS NEGATIVO = Azienda in perdita

💡 **Esempio:**
Microsoft: utile €20 miliardi, 7.5 miliardi di azioni
EPS = 20 / 7.5 = **€2.67 per azione**

🔍 **Come usarlo:**
Confronta EPS anno su anno:
- 2022: EPS €2.00
- 2023: EPS €2.50
→ Crescita del 25% = OTTIMO SEGNALE!
        """,
        "keywords": ["eps", "earnings per share", "utile", "azione", "guadagno"]
    },

    # --- NUOVI CONCETTI DI COMPLEXITY SCIENCE ---

    "q_score": {
        "name": "Q-Score (Quantum Score)",
        "explanation": """
**Q-Score** è l'indicatore proprietario della Quantum Terminal (0-100).
Sostituisce il vecchio 'SmartQuant'.

🧠 **Come viene calcolato:**
Combina analisi Tecnica, Fondamentale, Momentum e Qualità in un unico numero.
- **80-100** = STRONG BUY (Eccellente)
- **60-79** = BUY (Buono)
- **40-59** = HOLD (Neutrale)
- **20-39** = SELL (Debole)
- **0-19** = STRONG SELL (Pessimo)

Usa questo score per filtrare rapidamente gli asset migliori!
        """,
        "keywords": ["q-score", "q score", "score", "smartquant", "rating", "punteggio"]
    },

    "hurst": {
        "name": "Esponente di Hurst",
        "explanation": """
**L'Esponente di Hurst (H)** misura la 'memoria' di una serie temporale.
Viene dalla Teoria del Caos e dai Frattali.

📐 **Valori:**
- **H = 0.5** = Random Walk (Imprevedibile, come il lancio di una moneta).
- **H > 0.5** = **Trend Persistente** (Se sale, tende a continuare a salire).
- **H < 0.5** = **Mean Reverting** (Se sale troppo, tende a tornare indietro).

💡 **Applicazione:**
- Se H=0.7 su Bitcoin: Il trend è forte, segui il momentum!
- Se H=0.3 su EUR/USD: Il mercato è laterale, compra basso e vendi alto (swing trading).
        """,
        "keywords": ["hurst", "esponente", "caos", "frattale", "memoria", "persistenza"]
    },

    "pareto": {
        "name": "Legge di Pareto & Gini",
        "explanation": """
**Principio di Pareto (80/20):** L'80% della ricchezza è detenuta dal 20% della popolazione.
È universale nei sistemi complessi.

**Indice di Gini:** Misura la disuguaglianza (0 = uguaglianza perfetta, 1 = massima disuguaglianza).

🌍 **In Macroeconomia:**
- Gini alto (> 0.50) = Rischio instabilità sociale.
- Pareto estremo = Pochi attori controllano il mercato (es. Big Tech nell'S&P 500).

Se pochi titoli guidano tutto il rialzo, il mercato è fragile ("Narrow Breadth").
        """,
        "keywords": ["pareto", "gini", "ricchezza", "disuguaglianza", "distribuzione", "elite"]
    },

    "entropy": {
        "name": "Entropia di Shannon",
        "explanation": """
**L'Entropia di Shannon** misura l'informazione o l'incertezza in un sistema.

Maggiore è l'entropia, più il mercato è "sorprendente" e caotico.
Bassa entropia significa che il mercato è ordinato e prevedibile.

📉 **Crollo di mercato:** Spesso l'entropia diminuisce improvvisamente prima di un crash, perché tutti iniziano a fare la stessa cosa (vendere = alta correlazione = bassa entropia).
        """,
        "keywords": ["entropia", "shannon", "caos", "incertezza", "informazione"]
    },

    "network_topology": {
        "name": "Network Topology & PageRank",
        "explanation": """
Applichiamo la teoria delle reti al commercio globale.

🕸️ **Trade PageRank:**
Identifica i "nodi critici" nell'economia.
- Un paese non è importante solo per il PIL, ma per **chi** commercia con lui.
- Se la Cina (Hub manifatturiero) si ferma, blocca tutta la rete.

**Clustering:**
Misura quanto sono interconnessi i settori (AI, Difesa, Crypto).
Alto clustering = Resilienza ma rischio contagio sistemico.
        """,
        "keywords": ["network", "rete", "pagerank", "topology", "nodi", "cluster", "approvvigionamento"]
    },

    "general": {
        "name": "Guida Generale ai Grafici",
        "explanation": """
🎓 **COME LEGGERE I GRAFICI NELL'APP**

📈 **TAB ANALYSIS:**
- **Grafico Prezzo** = Mostra andamento ultimi 90 giorni
- **Linea Cyan** = Prezzo reale
- **Linea Purple tratteggiata** = SMA 20 (trend breve)
- **Linea Verde puntini** = SMA 50 (trend medio)

🧬 **TAB PORTFOLIO OPTIMIZATION:**
- **Heatmap** = Mostra correlazioni (verde = buono)
- **Scatter Plot** = Ogni punto = portfolio, stella = ottimale

🌍 **TAB MACRO & COMPLEXITY:**
- **Complexity Science** = Nuovi indicatori avanzati (Hurst, Entropia, Reti)
- **Wealth Distribution** = Grafici Pareto/Gini
- **Global Network** = Visualizzazione interconnessioni commerciali

🎯 **SEGNALI:**
- 🟢 BUY = Segnale di acquisto
- 🔴 SELL = Segnale di vendita
- 🟡 HOLD = Attendere
        """,
        "keywords": ["grafico", "chart", "come leggere", "guida", "help", "aiuto", "spiegazione"]
    }
}

def get_ai_response(user_query):
    """
    Get AI response based on user query
    Returns best matching explanation or general help
    """
    query_lower = user_query.lower()
    
    # Score each topic based on keyword matches
    scores = {}
    for topic_key, topic_data in AI_KNOWLEDGE_BASE.items():
        score = 0
        for keyword in topic_data["keywords"]:
            if keyword in query_lower:
                score += 1
        scores[topic_key] = score
    
    # Get best match
    best_match = max(scores.items(), key=lambda x: x[1])
    
    if best_match[1] > 0:
        # Found relevant topic
        topic = AI_KNOWLEDGE_BASE[best_match[0]]
        return f"### {topic['name']}\n\n{topic['explanation']}"
    else:
        # No match, return general help
        return """
### 🤖 Quantum AI Assistant

Non ho trovato una risposta specifica. Ecco cosa posso spiegarti:

**📊 Indicatori Tecnici:**
- RSI, SMA, Volatilità, Sharpe Ratio

**🕸️ Complexity Science (NUOVO):**
- **Q-Score** (ex SmartQuant)
- Esponente di Hurst (Caos & Trend)
- Network Topology (PageRank)
- Entropia di Shannon
- Legge di Pareto & Gini

**💰 Metriche Fondamentali:**
- P/E Ratio, Market Cap, EPS, Dividend Yield

**💡 Esempi di domande:**
- "Cos'è il Q-Score?"
- "Spiega l'esponente di Hurst"
- "A cosa serve l'entropia nei mercati?"
- "Come funziona il PageRank nel commercio?"
- "Cos'è la legge di Pareto?"

Fai una domanda più specifica! 🚀
        """

def get_chart_tips():
    """Get quick tips for reading charts"""
    return """
### 📈 Quick Chart Reading Tips

**Linee sul grafico:**
- 🔵 **Cyan** = Prezzo attuale
- 🟣 **Purple tratteggiata** = Media 20 giorni
- 🟢 **Verde puntini** = Media 50 giorni

**Complexity Metrics:**
- **Hurst > 0.5** = Trend Forte (Segui il movimento)
- **Hurst < 0.5** = Mean Reverting (Aspettati un ritorno alla media)
    """

def get_fundamental_tips():
    """Get tips for fundamental analysis"""
    return """
### 💼 Fundamental Analysis Basics

**Azioni (Stocks):**
- **Q-Score > 80** = Strong Buy (Segnale molto forte)
- **P/E basso** = Potenzialmente sottovalutato
- **EPS in crescita** = Azienda in salute

**Complexity:**
- **Alto Network PageRank** = "Too Big To Fail" (Settore critico)
- **Alta Entropia** = Mercato confuso/incerto
    """

