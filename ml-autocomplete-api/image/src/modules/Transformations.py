import pandas

from typing import List


class Transformations:
    def __init__():
        pass

    def strip_text(text: str):
        return text.strip()

    def encode_expert_diagnose(feature: pandas.Series) -> pandas.Series:
        return feature.map(
            {"Normal": 0, "Bipolar Type-1": 1, "Bipolar Type-2": 2, "Depression": 3}
        ).astype(int)

    def encode_binary_categories(
        df: pandas.DataFrame, binary_categories: List[str]
    ) -> pandas.DataFrame:
        for category in binary_categories:
            df[category] = df[category].map({"YES": 1, "NO": 0}).astype(int)
        return df
