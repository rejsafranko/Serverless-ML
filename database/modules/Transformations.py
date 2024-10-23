import pandas

from typing import List


class Transformations:
    def __init__(self):
        pass

    def strip_text(text: str):
        return text.strip()

    def encode_labels(feature: pandas.Series) -> pandas.Series:
        return feature.map(
            {"Normal": 0, "Bipolar Type-1": 1, "Bipolar Type-2": 2, "Depression": 3}
        ).astype(int)

    def encode_binary_categories(self, df: pandas.DataFrame) -> pandas.DataFrame:
        for category in df.columns:
            unique_values = df[category].dropna().unique()
            if len(unique_values) == 2:
                df[category] = (
                    df[category]
                    .apply(lambda x: x.lower())
                    .map({"yes": 1, "no": 0})
                    .astype(int)
                )
        return df

    def apply_all(self, df: pandas.DataFrame) -> pandas.DataFrame:
        df = df.apply(self.strip_text)
        df = self.encode_binary_categories(df)
        df["Expert Diagnose"] = df["Expert Diagnose"].apply(self.encode_labels)
        return df
