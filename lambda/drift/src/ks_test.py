import pandas
import scipy
import scipy.stats


def split_reference_current(
    df: pandas.DataFrame,
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    midpoint = len(df) // 2
    reference = df.iloc[:midpoint].reset_index(drop=True)
    current = df.iloc[midpoint:].reset_index(drop=True)
    return reference, current


def run_ks_tests(
    reference: pandas.DataFrame, current: pandas.DataFrame
) -> dict[str, float]:
    ks_stats = {}
    for col in reference.columns:
        if col in ("id", "created_at"):
            continue
        stat, _ = scipy.stats.ks_2samp(reference[col], current[col])
        ks_stats[col] = round(float(stat), 4)
    return ks_stats
