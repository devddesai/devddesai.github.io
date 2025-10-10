---
title: 'QuantChallenge 2025'
date: 2025-10-03T18:47:40-06:00
author: ["Dev Desai"]
draft: false
categories: ["notes"]
tags: ["Quantitative Finance", "Python", "Machine Learning"]
showtoc: true
math: true
---

## Simple Statistical Modeling

- Rolling window for average and volatility

## Monte Carlo

- use $\mu$ and $\sigma$ to simulate thousand of possible game outcomes
- probaility of winning into a fair value price
- adjust our fair value price with the tru emarket price. we assume we have some edge to the market.
- Real fair = 0.7 our fair + 0.3 market fair

## Market Making

- Provide liquidity to the market assuming the fair value and that people will want to trade at that fair value. 
- lets say fair value is 70
- post buy orders slightly below, so 69
- post sell orders slightly above, so 71
- bid ask spread of 2
- QA of some questions that i initially had
    - Q: Why not buy or sell far from the fair value?
    - A: Because extreme prices are usually stale or informed, increase risk, and can trap you with bad inventory. people won't usually buy so dont put liquidity into stale stocks when instead it can be used to turn more profit
    - Q: Why not buy something that’s super cheap?
    - A: “Cheap” prices often mean you’re late to new bad info, or the quote is gone before you reach it; chasing them adds risk.
    - Q: What is the purpose of market making?
    - A: To provide liquidity— always ready to buy/sell so others can trade instantly — and earn the small spread for doing so. for a buyer and seller to happen to have the same exact price ask at a moment is rare, which is why market makers are there

## Skew Safety Net

Terminology: 
- post a sell, or *ask*, someone else buys from you
- post a buy, or *bid*, someone else sells to you

If we use the market making rule from above, it is possible that we end up in a long position(have bought too much, meaning we are betting on the price increasing in the future) or a short position (have sold too much, meaning we are betting on the price decreasing in the future). If we end up in too much of a long or short position, that is risky. Again a market making scheme isn't to gain profit from large bets on prices, but rather the convenience fee of liquidity. To protect us from this, we can implement the following:

**Scenario 1**: Long
If we have been buying too much, we have a large inventory. To reduce it, we move our ask price down, making it easier for others to buy. We move our bid price down too, making it less likely that we buy more. 

**Scenario 2**: Short
The exact opposite

Still, there is still risk of having an extreme position, so we have a maximum position limit which will have a hard limit on if you can buy/sell

## Capital and PnL Tracking
Algorithm keeps track of PnL: current unrealized + realized profit
also capital_remaining: how much buying power we have left
Used to stop trading if capital runs too low and analyze performance.
PnL = (capital remaning - initial capital) + position * mid price


