import pandas


class Transformations:
    @staticmethod
    def strip_text(text: str) -> str:
        """
        Strips leading and trailing spaces from a string.
        """
        return text.strip()

    @staticmethod
    def encode_labels(feature: pandas.Series) -> pandas.Series:
        """
        Encodes categorical labels into integers.
        """
        return feature.map(
            {"Normal": 0, "Bipolar Type-1": 1, "Bipolar Type-2": 2, "Depression": 3}
        ).astype(int)

    @staticmethod
    def encode_binary_categories(dataframe: pandas.DataFrame) -> pandas.DataFrame:
        """
        Encodes binary categorical features ("yes"/"no") as 1/0.
        """
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

    @staticmethod
    def apply_all(dataframe: pandas.DataFrame) -> pandas.DataFrame:
        """
        Applies all transformations to the DataFrame.
        """
        dataframe = dataframe.apply(Transformations.strip_text)
        dataframe = Transformations.encode_binary_categories(dataframe)
        dataframe["Expert Diagnose"] = dataframe["Expert Diagnose"].apply(
            Transformations.encode_labels
        )
        return dataframe
