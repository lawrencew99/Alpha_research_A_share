from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from ashare_quant.data.cache import ParquetCache
from ashare_quant.data.demo import make_demo_data


class AKShareClient:
    """AKShare adapter with local parquet cache and demo fallback."""

    def __init__(self, cache_dir: str | Path):
        self.cache = ParquetCache(cache_dir)

    def fetch_universe(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
        demo: bool = False,
        fallback_demo: bool = True,
    ) -> dict[str, pd.DataFrame]:
        if demo:
            return make_demo_data(symbols, start_date, end_date)
        try:
            return self._fetch_from_akshare(symbols, start_date, end_date, adjust)
        except Exception as exc:  # pragma: no cover - depends on network and AKShare schema.
            if not fallback_demo:
                raise
            logger.warning("AKShare 数据获取失败，使用模拟数据兜底: {}", exc)
            return make_demo_data(symbols, start_date, end_date)

    def _fetch_from_akshare(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> dict[str, pd.DataFrame]:
        import akshare as ak

        price_frames = []
        fundamental_frames = []
        for symbol in symbols:
            cache_key = f"{symbol}_{start_date}_{end_date}_{adjust}"
            price = self.cache.read("prices", cache_key)
            if price is None:
                price = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust=adjust,
                )
                price = self._normalize_price(price, symbol)
                self.cache.write("prices", cache_key, price)
            price_frames.append(price)

            fundamentals = self.cache.read("fundamentals", cache_key)
            if fundamentals is None:
                fundamentals = self._fetch_fundamentals(ak, symbol, price)
                self.cache.write("fundamentals", cache_key, fundamentals)
            fundamental_frames.append(fundamentals)

        industries = self._fetch_industries(symbols)
        return {
            "prices": pd.concat(price_frames, ignore_index=True),
            "fundamentals": pd.concat(fundamental_frames, ignore_index=True),
            "industries": industries,
        }

    @staticmethod
    def _normalize_price(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        rename = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
            "涨跌幅": "pct_change",
        }
        frame = data.rename(columns=rename).copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["symbol"] = symbol
        frame["is_suspended"] = frame["volume"].fillna(0).eq(0)
        frame["limit_up"] = frame.groupby("symbol")["close"].shift(1) * 1.1
        frame["limit_down"] = frame.groupby("symbol")["close"].shift(1) * 0.9
        columns = [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "turnover",
            "is_suspended",
            "limit_up",
            "limit_down",
        ]
        return frame[[column for column in columns if column in frame.columns]]

    @staticmethod
    def _fetch_fundamentals(ak_module, symbol: str, price: pd.DataFrame) -> pd.DataFrame:
        base = price[["date", "symbol"]].copy()
        try:
            spot = ak_module.stock_a_indicator_lg(symbol=symbol)
            spot = spot.rename(
                columns={"trade_date": "date", "pe_ttm": "pe_ttm", "pb": "pb", "total_mv": "market_cap"}
            )
            spot["date"] = pd.to_datetime(spot["date"])
            keep = [column for column in ["date", "pe_ttm", "pb", "market_cap"] if column in spot]
            base = base.merge(spot[keep], on="date", how="left")
        except Exception as exc:  # pragma: no cover
            logger.warning("{} 估值数据获取失败，将使用价格衍生占位字段: {}", symbol, exc)
        base["market_cap"] = base.get("market_cap", base.index.to_series() * 0 + 1e10)
        base["pe_ttm"] = base.get("pe_ttm", 20.0)
        base["pb"] = base.get("pb", 2.0)
        base["roe"] = 1 / base["pb"].replace(0, pd.NA)
        base["debt_to_asset"] = 0.45
        return base.ffill().bfill()

    @staticmethod
    def _fetch_industries(symbols: list[str]) -> pd.DataFrame:
        industries = ["金融", "消费", "医药", "科技", "周期"]
        return pd.DataFrame(
            {"symbol": symbol, "industry": industries[index % len(industries)]}
            for index, symbol in enumerate(symbols)
        )
