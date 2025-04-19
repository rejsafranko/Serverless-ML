import pandas


class Transformations:
    @staticmethod
    def strip_text(text: str) -> str:
        return text.strip() if isinstance(text, str) else text

    @staticmethod
    def encode_labels(series: pandas.Series) -> pandas.Series:
        mapping = {
            "Normal": 0,
            "Bipolar Type-1": 1,
            "Bipolar Type-2": 2,
            "Depression": 3,
        }
        return series.map(mapping).astype(int)

    @staticmethod
    def encode_binary_categories(df: pandas.DataFrame) -> pandas.DataFrame:
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 2:
                df[col] = (
                    df[col]
                    .apply(lambda x: str(x).lower().strip())
                    .map({"yes": 1, "no": 0})
                    .astype(int)
                )
        return df

    @staticmethod
    def apply_all(df: pandas.DataFrame) -> pandas.DataFrame:
        df = df.apply(Transformations.strip_text)
        df = Transformations.encode_binary_categories(df)
        if "Expert_Diagnose" in df.columns:
            df["Expert_Diagnose"] = Transformations.encode_labels(df["Expert_Diagnose"])
        return df
