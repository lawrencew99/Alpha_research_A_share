from __future__ import annotations

from pathlib import Path

import pandas as pd


class ParquetCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, namespace: str, key: str) -> Path:
        safe_key = key.replace("/", "_").replace("\\", "_").replace(":", "-")
        folder = self.root / namespace
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{safe_key}.parquet"

    def read(self, namespace: str, key: str) -> pd.DataFrame | None:
        path = self.path(namespace, key)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def write(self, namespace: str, key: str, data: pd.DataFrame) -> Path:
        path = self.path(namespace, key)
        data.to_parquet(path, index=False)
        return path
