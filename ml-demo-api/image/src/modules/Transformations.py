import pandas


class Transformations:
    def __init__(self) -> None:
        pass

    def strip_text(text: str) -> str:
        return text.strip()

    def encode_labels(feature: pandas.Series) -> pandas.Series:
        return feature.map(
            {"Normal": 0, "Bipolar Type-1": 1, "Bipolar Type-2": 2, "Depression": 3}
        ).astype(int)

    def encode_binary_categories(self, dataframe: pandas.DataFrame) -> pandas.DataFrame:
        for category in dataframe.columns:
            unique_values = dataframe[category].dropna().unique()
            if len(unique_values) == 2:
                dataframe[category] = (
                    dataframe[category]
                    .apply(lambda x: x.lower())
                    .map({"yes": 1, "no": 0})
                    .astype(int)
                )
        return dataframe

    def apply_all(self, dataframe: pandas.DataFrame) -> pandas.DataFrame:
        dataframe = dataframe.apply(self.strip_text)
        dataframe = self.encode_binary_categories(dataframe)
        dataframe["Expert Diagnose"] = dataframe["Expert Diagnose"].apply(
            self.encode_labels
        )
        return dataframe
