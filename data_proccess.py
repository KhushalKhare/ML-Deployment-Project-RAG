import pandas as pd
from pathlib import Path

RAW = Path("data/sample/Actual_generation_202512010000_202601010000_Hour.csv")
OUT = Path("data/processed/generation_de_hourly.csv")

def main():
    df = pd.read_csv(
        RAW,
        sep=";",
        na_values=["-", "–", "", " "],   # common “not available” markers
        keep_default_na=True
    )

    # Timestamp
    df["timestamp"] = pd.to_datetime(df["Start date"], utc=True)

    # All generation columns
    value_cols = [c for c in df.columns if "[MWh]" in c]

    # Convert to numeric safely
    for col in value_cols:
        s = df[col].astype(str).str.strip()

        # handle formatting variants:
        # - remove thousands separators if present
        # - normalize decimal comma to dot if present
        s = s.str.replace(".", "", regex=False)      # thousands separator (if used)
        s = s.str.replace(",", ".", regex=False)     # decimal comma -> dot
        s = s.replace({"nan": None, "None": None, "-": None, "–": None})

        df[col] = pd.to_numeric(s, errors="coerce")

    # Total generation (skip NaNs)
    df["value"] = df[value_cols].sum(axis=1, skipna=True)

    out = df[["timestamp", "value"]].sort_values("timestamp")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    # quick sanity stats
    print(f"Saved {len(out)} rows to {OUT}")
    print("Missing value rows:", int(out["value"].isna().sum()))
    print("Min/Max:", float(out["value"].min()), float(out["value"].max()))

if __name__ == "__main__":
    main()
