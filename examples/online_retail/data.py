"""
UCI Online Retail dataset for session-based next-item recommendation.

Each row in the source file is a single line-item of a purchase
(invoice). We treat the ``InvoiceNo`` as a session identifier — no stable
``user_id`` is required, which is the pedagogical point of this sample:
the framework's ``user_column`` is happy to be any indexable entity,
including an ephemeral session.

Dataset: https://archive.ics.uci.edu/ml/datasets/online+retail
- ~540K line-items, ~25K invoices (sessions), ~4K stock codes (items)
- Dec 2010 → Dec 2011

This was chosen over YooChoose / RetailRocket because both of those
require Kaggle credentials — UCI hosts Online Retail openly.
"""

import os
from typing import Optional

import luigi
import numpy as np
import pandas as pd

from mars_rec.data.task import BasePrepareDataFrames

OUTPUT_PATH: str = os.environ.get("OUTPUT_PATH", "output")
BASE_DIR: str = os.path.join(OUTPUT_PATH, "online_retail")
DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")

ONLINE_RETAIL_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

HISTORY_LEN: int = 8  # per-session history window
MIN_SESSION_LEN: int = 2  # need at least 2 items to produce 1 next-item row
MIN_ITEM_COUNT: int = 5  # drop very rare items
PAD_TOKEN: str = "-1"


class DownloadAndPrepareOnlineRetail(luigi.Task):
    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "online_retail_sessions.csv"))

    def run(self):
        import math

        import requests
        from tqdm import tqdm

        os.makedirs(DATASET_DIR, exist_ok=True)

        xlsx_path = os.path.join(DATASET_DIR, "Online Retail.xlsx")
        if not os.path.exists(xlsx_path):
            print(f"Downloading Online Retail from {ONLINE_RETAIL_URL}...")
            r = requests.get(ONLINE_RETAIL_URL, stream=True)
            total = int(r.headers.get("content-length", 0))
            block = 1024
            with open(xlsx_path, "wb") as f:
                for chunk in tqdm(
                    r.iter_content(block),
                    total=math.ceil(total / block) if total else 0,
                    unit="KB",
                    unit_scale=True,
                ):
                    f.write(chunk)

        print("Reading xlsx (this takes ~30s)...")
        df = pd.read_excel(xlsx_path, engine="openpyxl")

        df.columns = [c.strip() for c in df.columns]
        df = df.dropna(subset=["InvoiceNo", "StockCode", "InvoiceDate"])
        # Drop credit notes (InvoiceNo starting with 'C') and non-purchases
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
        df = df[df["Quantity"] > 0]
        df["StockCode"] = df["StockCode"].astype(str)
        df["InvoiceNo"] = df["InvoiceNo"].astype(str)

        item_counts = df["StockCode"].value_counts()
        keep_items = item_counts[item_counts >= MIN_ITEM_COUNT].index
        df = df[df["StockCode"].isin(keep_items)]

        # Build a within-session order using InvoiceDate as the primary key
        # and the source-file row order as the tiebreaker (line-items in an
        # invoice share a date stamp).
        df = df.reset_index(drop=False).rename(columns={"index": "_row_order"})
        df["timestamp"] = (pd.to_datetime(df["InvoiceDate"]).astype("int64") // 1_000_000_000).astype(np.int64)
        df = df.sort_values(["InvoiceNo", "timestamp", "_row_order"]).reset_index(drop=True)

        session_lens = df.groupby("InvoiceNo")["StockCode"].size()
        keep_sessions = session_lens[session_lens >= MIN_SESSION_LEN].index
        df = df[df["InvoiceNo"].isin(keep_sessions)]

        rows = []
        for _, group in df.groupby("InvoiceNo", sort=False):
            prior: list = []
            for invoice, code, ts in zip(
                group["InvoiceNo"].values,
                group["StockCode"].values,
                group["timestamp"].values,
            ):
                if prior:
                    padded = [PAD_TOKEN] * (HISTORY_LEN - len(prior)) + [str(c) for c in prior[-HISTORY_LEN:]]
                    rows.append((str(invoice), str(code), int(ts), padded))
                prior.append(code)

        out = pd.DataFrame(
            rows,
            columns=["session_id", "stock_code", "timestamp", "history_stock_codes"],
        )
        out["label"] = 1.0
        out["history_stock_codes"] = out["history_stock_codes"].apply(
            lambda lst: "[" + ", ".join(f"'{v}'" for v in lst) + "]"
        )

        out.to_csv(self.output().path, index=False)
        print(f"Online Retail session dataset prepared: {len(out)} next-item rows")
        print(f"  Sessions: {out['session_id'].nunique()}")
        print(f"  Items: {out['stock_code'].nunique()}")


class PrepareOnlineRetailDataFrame(BasePrepareDataFrames):
    def requires(self):
        return DownloadAndPrepareOnlineRetail()

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def read_data_frame_path(self) -> str:
        return self.input().path

    @property
    def metadata_data_frame_path(self) -> Optional[str]:
        return None

    @property
    def timestamp_property(self) -> Optional[str]:
        return "timestamp"

    def create_available_items(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        return df
