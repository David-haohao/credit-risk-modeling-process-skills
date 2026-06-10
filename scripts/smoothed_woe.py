"""Additively smoothed WOE transformation fitted on training data."""

from __future__ import annotations

import numpy as np
import pandas as pd


MISSING_VALUE = "__MISSING__"


def _normalized(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), MISSING_VALUE)


class SmoothedWOETransformer:
    """Apply one consistent additive smoothing formula to WOE and IV."""

    def __init__(self, smooth: float = 0.5):
        if smooth <= 0:
            raise ValueError("smooth 必须大于 0")
        self.smooth = smooth
        self.mappings: dict[str, dict[object, float]] = {}
        self.iv_values: dict[str, float] = {}

    def fit(self, data: pd.DataFrame, target: pd.Series) -> "SmoothedWOETransformer":
        target = pd.Series(target, index=data.index)
        n_bad_total = int((target == 1).sum())
        n_good_total = int((target == 0).sum())
        if n_bad_total == 0 or n_good_total == 0:
            raise ValueError("WOE 拟合数据必须同时包含 good 和 bad 样本")

        for column in data.columns:
            values = _normalized(data[column])
            table = pd.crosstab(values, target).reindex(columns=[0, 1], fill_value=0)
            n_bins = len(table)
            good_dist = (table[0] + self.smooth) / (n_good_total + self.smooth * n_bins)
            bad_dist = (table[1] + self.smooth) / (n_bad_total + self.smooth * n_bins)
            woe = np.log(bad_dist / good_dist)
            self.mappings[column] = woe.to_dict()
            self.iv_values[column] = float(((bad_dist - good_dist) * woe).sum())

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed = pd.DataFrame(index=data.index)
        for column in data.columns:
            mapping = self.mappings[column]
            values = _normalized(data[column])
            result = values.map(mapping)
            if result.isna().any():
                unknown = values[result.isna()].drop_duplicates().tolist()
                raise ValueError(f"{column} 出现训练分箱规则之外的值: {unknown}")
            transformed[column] = result.astype(float)
        return transformed

    def fit_transform(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        return self.fit(data, target).transform(data)

    def iv_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.iv_values.items(), columns=["column", "iv"])
