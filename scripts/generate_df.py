from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
MAIN_DIRS = ["train", "valid", "test"]


def main():
    df = pd.DataFrame()
    for folder in MAIN_DIRS:
        data_path = DATA_DIR / folder
        data_pathes = data_path.glob("**/*")
        data_pathes = [f for f in data_pathes if f.name.endswith("jpg")]
        inner_df = pd.DataFrame({"image_path": data_pathes})
        inner_df["split"] = folder
        df = pd.concat([df, inner_df], ignore_index=True)

    df.to_csv(DATA_DIR / "data.csv")


if __name__ == "__main__":
    main()
