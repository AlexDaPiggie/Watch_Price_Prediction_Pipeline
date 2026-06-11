import os
import re
import shutil
import warnings

import kagglehub
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def fetch():
    if not os.path.exists("Watches.csv"):
        data_path = kagglehub.dataset_download("philmorekoung11/luxury-watch-listings")
        source = os.path.join(data_path, "Watches.csv")
        destination = os.getcwd()
        shutil.copy2(source, destination)
    return os.path.join(os.getcwd(), "Watches.csv")


def load_data(path):
    return pd.read_csv(path)


def first_mode(series):
    mode = series.dropna().mode()
    return mode.iloc[0] if not mode.empty else np.nan


def parse_price_series(price_series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        price_series.replace({r"\$": "", ",": ""}, regex=True),
        errors="coerce",
    )


def make_price_strata(price_series: pd.Series, n_bins=10) -> pd.Series:
    price_numeric = parse_price_series(price_series)
    max_bins = min(n_bins, price_numeric.nunique(), len(price_numeric))

    if max_bins < 2:
        return pd.Series("all_prices", index=price_series.index)

    return pd.qcut(
        price_numeric,
        q=max_bins,
        duplicates="drop",
    ).astype(str)


def split_train_test_by_price(
    df: pd.DataFrame,
    test_size=0.2,
    random_state=RANDOM_STATE,
    n_bins=10,
):
    price_numeric = parse_price_series(df["price"])
    valid_price = price_numeric.notna()
    df = df.loc[valid_price].copy()
    price_numeric = price_numeric.loc[valid_price]

    n_rows = len(df)
    n_test = int(np.ceil(n_rows * test_size)) if isinstance(test_size, float) else test_size
    n_train = n_rows - n_test
    usable_bins = min(n_bins, price_numeric.nunique(), n_train, n_test)
    stratify = None

    if usable_bins >= 2:
        stratify = make_price_strata(price_numeric, n_bins=usable_bins)

    return train_test_split(
        df,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )


class WatchDataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.year_pattern = r"(15[5-9]\d|1[6-9]\d{2}|200\d|201\d|202[0-4])"
        self.cond_map = {
            "Unworn": "new",
            "Very good": "good",
            "New": "good",
            "Good": "good",
            "Poor": "bad",
            "Fair": "bad",
            "Incomplete": "bad",
        }
        self.female_keywords = [
            "lady",
            "ladies",
            "women",
            "womens",
            "woman",
            "mother of pearl",
            "mop",
        ]

        self.brand_set = None
        self.model_list = None
        self.yop_median_by_model = None
        self.mvmt_mode_by_ref = None
        self.mvmt_mode_by_modelD = None
        self.p65 = None
        self.p90 = None
        self.casem_map = None
        self.bracem_map = None
        self.casem_mode_by_ref = None
        self.bracem_mode_by_ref = None
        self.is_female_mode_by_ref = None
        self.is_female_mode_by_model = None
        self.is_female_mode_by_size = None
        self.cond_mode_by_pricebin_ref = None
        self.cond_mode_by_pricebin_model = None
        self.price_upper_cap = None
        self.price_lower_cap = None

    def fit(self, df: pd.DataFrame, y=None):
        df = df.copy()
        print ('Fitting Price...', end = ' ' )
        df = self.clean_price(df)
        print ('DONE!!!!')

        print ('Fitting Brand...', end = ' ' )
        self.fit_brand(df)
        df = self.clean_brand(df)
        print ('DONE!!!')

        print ('Fitting Yop...', end = ' ' )
        df = self.extract_yop(df)
        self.fit_yop(df)
        df = self.clean_yop(df)
        print ('DONE!!!')

        print ('Fitting Model... ', end = ' ')
        self.fit_model(df)
        df = self.clean_model(df)
        print ('DONE!!!')
    
        print ('Fitting Movement...', end = ' ')
        self.fit_mvmt(df)
        df = self.clean_mvmt(df)
        print ('DONE!!!')

        print ('Fitting Casem and Bracem...', end = ' ')
        self.fit_materials(df)
        df = self.clean_casem_and_bracem(df)
        print ('DONE!!!')

        print ('Fitting Sex...', end = ' ')
        self.fit_sex(df)
        df = self.clean_sex(df)
        print ('DONE!!!')

        print ('Fitting Cond...', end = ' ')
        self.fit_cond(df)
        df = self.clean_cond(df)
        print ('DONE!!!')

        return self

    def transform(self, df: pd.DataFrame):
        df = df.copy()

        print ('Transforming Price...', end = ' ')
        df = self.clean_price(df)
        print ('DONE!!!')

        print ('Transforming Brand...', end = ' ')
        df = self.clean_brand(df)
        print ('DONE!!!')
        
        print ('Transforming Yop...', end = ' ')
        df = self.extract_yop(df)
        df = self.clean_yop(df)
        print ('DONE!!!')

        print ('Transforming Model...', end = ' ')
        df = self.clean_model(df)
        print ('DONE!!!')

        print ('Transforming Movement...', end = ' ')
        df = self.clean_mvmt(df)
        print ('DONE!!!')

        print ('Transforming Casem and Bracem...', end = ' ')
        df = self.clean_casem_and_bracem(df)
        print ('DONE!!!')

        print ('Transforming Sex...', end = ' ')
        df = self.clean_sex(df)
        print ('DONE!!!')

        print ('Transforming Cond...', end = ' ')
        df = self.clean_cond(df)
        print ('DONE!!!')

        print ('Transforming Size...', end = ' ')
        df = self.clean_size(df)
        print ('DONE!!!')

        df = self.finalize(df)

        return df

    def clean_price(self, df: pd.DataFrame):
        df["price"] = df["price"].replace({r"\$": "", ",": ""}, regex=True)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price"])
        return df

    def fit_brand(self, df: pd.DataFrame):
        brand_list = df["brand"].dropna().unique().tolist()
        brand_set = set(brand_list)
        self.brand_set = sorted(
            [str(b) for b in brand_set if pd.notnull(b)],
            key=len,
            reverse=True,
        )

    def clean_brand(self, df: pd.DataFrame):
        def find_brand(name, original_brand):
            if str(original_brand) != "nan":
                return original_brand
            if pd.isnull(name):
                return original_brand
            for brand in self.brand_set:
                if brand.lower() in str(name).lower():
                    return brand
            return original_brand

        df["brand"] = df.apply(
            lambda row: find_brand(row["name"], row["brand"]),
            axis=1,
        )
        return df

    def extract_yop(self, df: pd.DataFrame):
        def extract_year(row):
            for col in ["yop", "name", "model"]:
                value = str(row[col])
                match = re.search(self.year_pattern, value)
                if match:
                    return int(match.group(1))
            return np.nan

        df["yop"] = df.apply(extract_year, axis=1)
        return df

    def fit_yop(self, df: pd.DataFrame):
        self.yop_median_by_model = df.groupby("model")["yop"].median()

    def clean_yop(self, df: pd.DataFrame) -> pd.DataFrame:
        df["yop"] = df["yop"].fillna(df["model"].map(self.yop_median_by_model))
        df = df.dropna(subset=["yop"])
        return df

    def fit_model(self, df: pd.DataFrame):
        model_list = df["model"].dropna().unique().tolist()
        self.model_list = [m for m in model_list if m != "Unknown"]

    def clean_model(self, df: pd.DataFrame) -> pd.DataFrame:
        def model_from_name(row):
            if pd.isnull(row["model"]):
                name_str = str(row["name"]).lower()
                for model in self.model_list:
                    if str(model).lower() in name_str:
                        return model
            return row["model"]

        df["model"] = df.apply(model_from_name, axis=1)
        df = df.dropna(subset=["model"])
        return df

    def fit_mvmt(self, df: pd.DataFrame):
        self.mvmt_mode_by_ref = df.groupby("ref")["mvmt"].agg(first_mode).to_dict()
        self.mvmt_mode_by_model = df.groupby("model")["mvmt"].agg(first_mode).to_dict()

    def clean_mvmt(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df["mvmt"].isna()
        df.loc[mask, "mvmt"] = df.loc[mask, "ref"].map(self.mvmt_mode_by_ref)

        def mvmt_from_name(row):
            if pd.isnull(row["mvmt"]):
                name_lower = str(row["name"]).lower()
                if any(
                    word in name_lower
                    for word in [
                        "quartz",
                        "battery",
                        "solar",
                        "solor-powered",
                        "hybrid",
                        "spring-drive",
                    ]
                ):
                    return "Quartz"
                if any(word in name_lower for word in ["automatic", "mechanical"]):
                    return "Automatic"
                if any(
                    word in name_lower
                    for word in [
                        "manual-winding",
                        "manual winding",
                        "self-winding",
                        "self winding",
                    ]
                ):
                    return "Manual winding"
            return row["mvmt"]

        df["mvmt"] = df.apply(mvmt_from_name, axis=1)

        mask = df["mvmt"].isna()
        df.loc[mask, "mvmt"] = df.loc[mask, "model"].map(self.mvmt_mode_by_model)
        df["mvmt"] = df["mvmt"].fillna("Unknown")
        return df

    def fit_materials(self, df: pd.DataFrame):
        self.p65 = df["price"].quantile(0.65)
        self.p90 = df["price"].quantile(0.90)

        casem_avg_price = df.groupby("casem")["price"].mean()
        bracem_avg_price = df.groupby("bracem")["price"].mean()

        def categorize_material(avg_price):
            if avg_price < self.p65:
                return "Common"
            if avg_price < self.p90:
                return "Valuable"
            return "Rare"

        self.casem_map = {
            material: categorize_material(price)
            for material, price in casem_avg_price.items()
        }
        self.bracem_map = {
            material: categorize_material(price)
            for material, price in bracem_avg_price.items()
        }

        temp_df = df.copy()
        temp_df["casem"] = temp_df["casem"].astype("string").str.strip()
        temp_df["bracem"] = temp_df["bracem"].astype("string").str.strip()
        temp_df["casem"] = temp_df["casem"].map(self.casem_map)
        temp_df["bracem"] = temp_df["bracem"].map(self.bracem_map)

        self.casem_mode_by_ref = temp_df.groupby("ref")["casem"].agg(first_mode).to_dict()
        self.bracem_mode_by_ref = temp_df.groupby("ref")["bracem"].agg(first_mode).to_dict()

    def clean_casem_and_bracem(self, df: pd.DataFrame) -> pd.DataFrame:
        df["casem"] = df["casem"].astype("string").str.strip()
        df["bracem"] = df["bracem"].astype("string").str.strip()

        df["casem"] = df["casem"].map(self.casem_map)
        df["bracem"] = df["bracem"].map(self.bracem_map)

        def material_from_name(row, col_name, material_map):
            value = row[col_name]
            if pd.isnull(value) or str(value).strip() == "":
                name_lower = str(row["name"]).lower()
                for keyword, material in material_map.items():
                    if pd.notnull(keyword) and str(keyword).lower() in name_lower:
                        return material
            return value

        df["casem"] = df.apply(
            lambda row: material_from_name(row, "casem", self.casem_map),
            axis=1,
        )
        df["bracem"] = df.apply(
            lambda row: material_from_name(row, "bracem", self.bracem_map),
            axis=1,
        )

        mask = df["casem"].isna()
        df.loc[mask, "casem"] = df.loc[mask, "ref"].map(self.casem_mode_by_ref)

        mask = df["bracem"].isna()
        df.loc[mask, "bracem"] = df.loc[mask, "ref"].map(self.bracem_mode_by_ref)

        df["casem"] = df["casem"].fillna("Unknown")
        df["bracem"] = df["bracem"].fillna("Unknown")
        return df

    def fit_sex(self, df: pd.DataFrame):
        temp_df = df.copy()
        temp_df["is_female"] = temp_df["sex"].map(
            {
                "Men's watch/Unisex": 0,
                "Women's watch": 1,
            }
        )
        temp_df["is_female"] = temp_df["is_female"].fillna(
            temp_df["name"].apply(self.female_from_name)
        )

        self.is_female_mode_by_ref = temp_df.groupby("ref")["is_female"].agg(first_mode).to_dict()
        self.is_female_mode_by_model = temp_df.groupby("model")["is_female"].agg(first_mode).to_dict()
        self.is_female_mode_by_size = temp_df.groupby("size")["is_female"].agg(first_mode).to_dict()

    def female_from_name(self, name):
        if pd.isna(name):
            return np.nan
        name_lower = str(name).lower()
        return 1 if any(word in name_lower for word in self.female_keywords) else 0

    def clean_sex(self, df: pd.DataFrame) -> pd.DataFrame:
        df["is_female"] = df["sex"].map(
            {
                "Men's watch/Unisex": 0,
                "Women's watch": 1,
            }
        )
        df["is_female"] = df["is_female"].fillna(df["name"].apply(self.female_from_name))

        mask = df["is_female"].isna()
        df.loc[mask, "is_female"] = df.loc[mask, "ref"].map(self.is_female_mode_by_ref)

        mask = df["is_female"].isna()
        df.loc[mask, "is_female"] = df.loc[mask, "model"].map(self.is_female_mode_by_model)

        mask = df["is_female"].isna()
        df.loc[mask, "is_female"] = df.loc[mask, "size"].map(self.is_female_mode_by_size)

        df = df.dropna(subset=["is_female"])

        if "sex" in df.columns:
            df = df.drop("sex", axis=1)

        return df

    def make_price_bins(self, price_series: pd.Series) -> pd.Series:
        return pd.cut(
            price_series,
            bins=[-np.inf, self.p65, self.p90, np.inf],
            labels=["Common", "Valuable", "Rare"],
        )

    def fit_cond(self, df: pd.DataFrame):
        temp_df = df.copy()
        temp_df["cond"] = temp_df["cond"].fillna(temp_df["condition"])
        temp_df["price_bins"] = self.make_price_bins(temp_df["price"])

        self.cond_mode_by_pricebin_ref = (
            temp_df.groupby(["price_bins", "ref"])["cond"].agg(first_mode).to_dict()
        )
        self.cond_mode_by_pricebin_model = (
            temp_df.groupby(["price_bins", "model"])["cond"].agg(first_mode).to_dict()
        )

    def clean_cond(self, df: pd.DataFrame) -> pd.DataFrame:
        df["cond"] = df["cond"].fillna(df["condition"])
        df["price_bins"] = self.make_price_bins(df["price"])

        mask = df["cond"].isna()
        df.loc[mask, "cond"] = [
            self.cond_mode_by_pricebin_ref.get((price_bin, ref), np.nan)
            for price_bin, ref in zip(df.loc[mask, "price_bins"], df.loc[mask, "ref"])
        ]

        mask = df["cond"].isna()
        df.loc[mask, "cond"] = [
            self.cond_mode_by_pricebin_model.get((price_bin, model), np.nan)
            for price_bin, model in zip(df.loc[mask, "price_bins"], df.loc[mask, "model"])
        ]

        df = df.dropna(subset=["cond"])
        df["cond"] = df["cond"].map(self.cond_map)
        return df

    def clean_size(self, df: pd.DataFrame) -> pd.DataFrame:
        df["size"] = df["size"].astype("string").str.lower()

        def extract_case_size(size):
            if pd.isna(size):
                return np.nan

            text = str(size).strip().lower().replace(",", ".").replace("`", "")

            match = re.search(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*mm", text)
            if match:
                value = max(float(match.group(1)), float(match.group(2)))
                return value if 3 <= value <= 55 else np.nan

            match = re.search(r"(\d+(?:\.\d+)?)\s*mm", text)
            if match:
                value = float(match.group(1))
                return value if 3 <= value <= 55 else np.nan

            return np.nan

        df["case_size_mm"] = df["size"].apply(extract_case_size)
        return df

    def finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["price_bins", "Unnamed: 0", "condition", "size"]:
            if col in df.columns:
                df = df.drop(col, axis=1)
        return df

def main():
    print (f"-"*16)
    print ('START CLEANING DATA')
    
    if os.path.exists('Watches_train_cleaned.csv') and os.path.exists('Watches_test_cleaned.csv'): 
        print ('Data Has Already Been Cleaned')
        print (f'Train: Watches_train_cleaned.csv')
        print (f'Test: Watches_test_cleaned.csv')
    
    else: 
        print ("Fetching data...", end = ' ')
        data_path = fetch()
        print ('DONE!!!')

        print ('Loading data...', end = ' ')
        df = load_data(data_path)
        print ('DONE!!!')
        
        train_df, test_df = split_train_test_by_price(
            df,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

        cleaner = WatchDataCleaner()

        train_cleaned = cleaner.fit_transform (train_df)
        print("Train Clean Data: DONE!!!")

        test_cleaned = cleaner.transform(test_df)
        print("Test Clean Data: DONE!!!")

        train_output_path = os.path.join(os.getcwd(), "Watches_train_cleaned.csv")
        test_output_path = os.path.join(os.getcwd(), "Watches_test_cleaned.csv")

        train_cleaned.to_csv(train_output_path, index=False)
        print ('Updated Watches_train_cleaned.csv')

        test_cleaned.to_csv(test_output_path, index=False)
        print ('Updated Watches_test_cleaned.csv')
    print ('FINISH CLEANING DATA')
    print (f"-"*16)

if __name__ == "__main__":
    main()
