from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_data(
    symbols: list[str],
    start_date: str,
    end_date: str,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, end_date)
    industries = ["金融", "消费", "医药", "科技", "周期"]
    price_rows: list[dict] = []
    fundamental_rows: list[dict] = []
    industry_rows: list[dict] = []

    for i, symbol in enumerate(symbols):
        drift = rng.normal(0.00025, 0.00015)
        vol = rng.uniform(0.012, 0.03)
        returns = rng.normal(drift, vol, len(dates))
        close = 20 * np.exp(np.cumsum(returns)) * (1 + i * 0.03)
        open_ = close * (1 + rng.normal(0, 0.003, len(dates)))
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, len(dates)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, len(dates)))
        volume = rng.lognormal(15, 0.4, len(dates)).astype(int)
        amount = volume * close
        turnover = rng.uniform(0.003, 0.05, len(dates))
        market_cap = rng.uniform(80, 2000) * 1e8 * close / close[0]
        pe = rng.uniform(8, 45, len(dates))
        pb = rng.uniform(0.8, 6, len(dates))
        roe = rng.uniform(0.03, 0.25, len(dates))
        debt_to_asset = rng.uniform(0.15, 0.75, len(dates))

        for j, date in enumerate(dates):
            price_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": open_[j],
                    "high": high[j],
                    "low": low[j],
                    "close": close[j],
                    "volume": volume[j],
                    "amount": amount[j],
                    "turnover": turnover[j],
                    "is_suspended": bool(rng.random() < 0.002),
                    "limit_up": close[j] * 1.1,
                    "limit_down": close[j] * 0.9,
                }
            )
            fundamental_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "market_cap": market_cap[j],
                    "pe_ttm": pe[j],
                    "pb": pb[j],
                    "roe": roe[j],
                    "debt_to_asset": debt_to_asset[j],
                }
            )
        industry_rows.append({"symbol": symbol, "industry": industries[i % len(industries)]})

    return {
        "prices": pd.DataFrame(price_rows),
        "fundamentals": pd.DataFrame(fundamental_rows),
        "industries": pd.DataFrame(industry_rows),
    }
