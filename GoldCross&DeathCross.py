"""
SMA Crossover Backtest
======================
Stratégie : on achète quand la SMA courte croise au-dessus de la SMA longue
            on vend quand la SMA courte repasse en dessous.
 
Auteur : [Ton nom]
"""
 
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
plt.style.use('dark_background') 
# ── Paramètres ──────────────────────────────────────────────────────────────
 
TICKER     = "AAPL"       # ETF S&P 500 — change en "BTC-USD", "AAPL", etc.
START      = "2018-01-01"
END        = "2024-01-01"
SMA_SHORT  = 50         # fenêtre courte (jours)
SMA_LONG   = 200        # fenêtre longue (jours)
CAPITAL    = 10_000      # capital de départ (€ ou $)
 
 
# ── 1. Téléchargement des données ────────────────────────────────────────────
 
def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = df[["Close"]].copy()
    df.columns = ["price"]
    return df
 
 
# ── 2. Calcul des indicateurs ─────────────────────────────────────────────────
 
def add_signals(df: pd.DataFrame, short: int, long: int) -> pd.DataFrame:
    df = df.copy()
    df["sma_short"] = df["price"].rolling(short).mean()
    df["sma_long"]  = df["price"].rolling(long).mean()
 
    # Signal : 1 = en position, 0 = hors marché
    df["signal"] = 0
    df.loc[df["sma_short"] > df["sma_long"], "signal"] = 1
 
    # On décale d'un jour pour éviter le look-ahead bias
    df["signal"] = df["signal"].shift(1)
 
    # Détection des croisements (trades)
    df["position"] = df["signal"].diff()  # +1 = achat, -1 = vente
    return df.dropna()
 
 
# ── 3. Calcul des performances ────────────────────────────────────────────────
 
def compute_performance(df: pd.DataFrame, capital: float) -> pd.DataFrame:
    df = df.copy()
 
    # Rendements journaliers du sous-jacent
    df["market_ret"] = df["price"].pct_change()
 
    # Rendements de la stratégie (on applique le signal du jour précédent)
    df["strat_ret"] = df["market_ret"] * df["signal"]
 
    # Valeur du portefeuille cumulée
    df["portfolio"]  = capital * (1 + df["strat_ret"]).cumprod()
    df["buy_hold"]   = capital * (1 + df["market_ret"]).cumprod()
 
    return df.dropna()
 
 
# ── 4. Métriques ──────────────────────────────────────────────────────────────
 
def metrics(df: pd.DataFrame) -> dict:
    strat_ret = df["strat_ret"]
    total_ret  = (df["portfolio"].iloc[-1] / df["portfolio"].iloc[0] - 1) * 100
    bh_ret     = (df["buy_hold"].iloc[-1]  / df["buy_hold"].iloc[0]  - 1) * 100
    sharpe     = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
 
    # Max Drawdown
    roll_max   = df["portfolio"].cummax()
    drawdown   = (df["portfolio"] - roll_max) / roll_max
    max_dd     = drawdown.min() * 100
 
    nb_trades  = int((df["position"] != 0).sum())
 
    return {
        "Rendement stratégie (%)": round(total_ret, 2),
        "Buy & Hold (%)":          round(bh_ret, 2),
        "Sharpe Ratio":            round(sharpe, 2),
        "Max Drawdown (%)":        round(max_dd, 2),
        "Nombre de trades":        nb_trades,
    }
 
 
# ── 5. Visualisation ──────────────────────────────────────────────────────────
 
def plot(df: pd.DataFrame, ticker: str, short: int, long: int):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"SMA Crossover Backtest — {ticker} ({short}/{long})", fontsize=14)
 
    # Graphique 1 : prix + SMAs + signaux
    ax1 = axes[0]
    ax1.plot(df.index, df["price"],     label="Prix",          color="#378ADD", linewidth=1)
    ax1.plot(df.index, df["sma_short"], label=f"SMA {short}",  color="#EF9F27", linewidth=1.2)
    ax1.plot(df.index, df["sma_long"],  label=f"SMA {long}",   color="#E24B4A", linewidth=1.2)
 
    # Signaux d'achat / vente
    buys  = df[df["position"] ==  1]
    sells = df[df["position"] == -1]
    ax1.scatter(buys.index,  buys["price"],  marker="^", color="#1D9E75", zorder=5, label="Achat",  s=60)
    ax1.scatter(sells.index, sells["price"], marker="v", color="#E24B4A", zorder=5, label="Vente",  s=60)
 
    ax1.legend(fontsize=9)
    ax1.set_ylabel("Prix ($)")
    ax1.grid(alpha=0.3)
 
    # Graphique 2 : performance cumulée
    ax2 = axes[1]
    ax2.plot(df.index, df["portfolio"], label="Stratégie",   color="#1D9E75", linewidth=1.5)
    ax2.plot(df.index, df["buy_hold"],  label="Buy & Hold",  color="#888780", linewidth=1, linestyle="--")
    ax2.legend(fontsize=9)
    ax2.set_ylabel("Valeur portefeuille ($)")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("backtest_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Graphique sauvegardé : backtest_result.png")
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print(f"Téléchargement de {TICKER}...")
    df = get_data(TICKER, START, END)
 
    df = add_signals(df, SMA_SHORT, SMA_LONG)
    df = compute_performance(df, CAPITAL)
 
    print("\n── Métriques ──────────────────────────────")
    for k, v in metrics(df).items():
        print(f"  {k:30s}: {v}")
    print("────────────────────────────────────────────\n")
 
    plot(df, TICKER, SMA_SHORT, SMA_LONG)
 