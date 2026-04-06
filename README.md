# Golden Cross & Death Cross — SMA Backtest

Backtest d'une stratégie de trading basée sur le croisement de moyennes mobiles simples (SMA).
Développé en Python dans le cadre de mon parcours vers le trading quantitatif.

## Stratégie

- **Golden Cross** : achat quand la SMA courte croise au-dessus de la SMA longue
- **Death Cross** : vente quand la SMA courte repasse en dessous de la SMA longue

## Métriques calculées

- Rendement total de la stratégie vs Buy & Hold
- Sharpe Ratio annualisé
- Max Drawdown
- Nombre de trades

## Stack

- Python 3.13
- pandas, numpy, matplotlib, yfinance

## Auteur

Zipettemelo — Étudiant L2 Mathématiques, Université Paris Cité  
Objectif : Trading Quantitatif