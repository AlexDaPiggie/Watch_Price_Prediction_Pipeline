import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


TARGET_COL = "price"
RANDOM_STATE = 42
CURRENT_YEAR = 2024

TRAIN_PATH = "Watches_train_cleaned.csv"
TEST_PATH = "Watches_test_cleaned.csv"
MODEL_OUTPUT_PATH = "Model/tree_pipeline.joblib"

BASE_NUMERIC_COLS = ["yop", "is_female", "case_size_mm"]
BASE_CATEGORICAL_COLS = ["brand", "model", "ref", "mvmt", "casem", "bracem", "cond"]
FINAL_CATEGORICAL_COLS = ["mvmt", "casem", "bracem", "cond", "brand"]

NAME_KEYWORDS = [
    "gold",
    "rose gold",
    "white gold",
    "yellow gold",
    "platinum",
    "diamond",
    "steel",
    "ceramic",
    "titanium",
    "chronograph",
    "limited",
    "vintage",
    "box",
    "papers",
    "full set",
    "unworn",
    "automatic",
    "quartz",
]

NAME_FEATURE_COLS = [
    "name_missing",
    "name_char_len",
    "name_word_count",
    "name_digit_count",
    "name_has_year",
] + [f"name_has_{keyword.replace(' ', '_')}" for keyword in NAME_KEYWORDS]

REF_STRUCTURE_COLS = [
    "ref_missing",
    "ref_char_len",
    "ref_digit_count",
    "ref_alpha_count",
    "ref_has_slash",
    "ref_has_dash",
    "ref_has_dot",
]

ENGINEERED_NUMERIC_COLS = BASE_NUMERIC_COLS + [
    "watch_age",
    "case_size_missing",
    "case_size_x_female",
    "age_x_case_size",
]

FREQUENCY_COLS = ["model", "ref", "brand_model_key", "model_ref_key"]
FREQUENCY_FEATURE_COLS = [f"{col}_frequency" for col in FREQUENCY_COLS]

OOF_TARGET_COLS = ["brand_model_key", "model", "ref"]
OOF_TARGET_SMOOTHING = {
    "brand_model_key": 20,
    "model": 20,
    "ref": 100,
}
OOF_TARGET_FEATURE_COLS = [f"{col}_target_mean" for col in OOF_TARGET_COLS]

FINAL_NUMERIC_COLS = (
    ENGINEERED_NUMERIC_COLS
    + FREQUENCY_FEATURE_COLS
    + NAME_FEATURE_COLS
    + REF_STRUCTURE_COLS
    + OOF_TARGET_FEATURE_COLS
)
FINAL_FEATURE_COLS = FINAL_NUMERIC_COLS + FINAL_CATEGORICAL_COLS

TUNED_EXTRA_TREE_PARAMS = {
    "n_estimators": 700,
    "min_samples_leaf": 1,
    "min_samples_split": 5,
    "max_features": 0.7108036432108452,
    "max_depth": None,
    "bootstrap": False,
    "random_state": RANDOM_STATE,
    "n_jobs": 3,
}


def load_cleaned_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def make_train_valid_split(df, valid_size=0.1, random_state=RANDOM_STATE):
    price_bins = pd.qcut(df[TARGET_COL], q=10, duplicates="drop")
    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        shuffle=True,
        stratify=price_bins,
        random_state=random_state,
    )
    return train_df.copy(), valid_df.copy()


def validate_required_columns(df, include_target=False):
    required = set(BASE_NUMERIC_COLS + BASE_CATEGORICAL_COLS + ["name"])
    if include_target:
        required.add(TARGET_COL)

    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def split_features_target(df):
    validate_required_columns(df, include_target=True)
    x = df.drop(columns=[TARGET_COL]).copy()
    y = np.log1p(df[TARGET_COL].astype(float))
    return x, y


class WatchTreeFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, current_year=CURRENT_YEAR):
        self.current_year = current_year

    def fit(self, x, y=None):
        validate_required_columns(x, include_target=False)
        return self

    def transform(self, x):
        validate_required_columns(x, include_target=False)
        df = x.copy()

        name = df["name"].fillna("").astype(str).str.lower()
        ref = df["ref"].fillna("").astype(str)

        df["watch_age"] = (
            self.current_year - pd.to_numeric(df["yop"], errors="coerce")
        ).clip(lower=0, upper=500)
        df["case_size_missing"] = df["case_size_mm"].isna().astype(int)
        df["case_size_x_female"] = pd.to_numeric(
            df["case_size_mm"],
            errors="coerce",
        ) * pd.to_numeric(df["is_female"], errors="coerce")
        df["age_x_case_size"] = df["watch_age"] * pd.to_numeric(
            df["case_size_mm"],
            errors="coerce",
        )

        df["name_missing"] = df["name"].isna().astype(int)
        df["name_char_len"] = name.str.len()
        df["name_word_count"] = name.str.split().str.len().fillna(0)
        df["name_digit_count"] = name.str.count(r"\d")
        df["name_has_year"] = name.str.contains(
            r"\b(?:19\d{2}|20[0-2]\d)\b",
            regex=True,
        ).astype(int)

        for keyword in NAME_KEYWORDS:
            col_name = f"name_has_{keyword.replace(' ', '_')}"
            df[col_name] = name.str.contains(keyword, regex=False).astype(int)

        df["ref_missing"] = df["ref"].isna().astype(int)
        df["ref_char_len"] = ref.str.len()
        df["ref_digit_count"] = ref.str.count(r"\d")
        df["ref_alpha_count"] = ref.str.count(r"[A-Za-z]")
        df["ref_has_slash"] = ref.str.contains("/", regex=False).astype(int)
        df["ref_has_dash"] = ref.str.contains("-", regex=False).astype(int)
        df["ref_has_dot"] = ref.str.contains(".", regex=False).astype(int)

        brand = df["brand"].fillna("missing").astype(str)
        model = df["model"].fillna("missing").astype(str)
        ref_value = df["ref"].fillna("missing").astype(str)
        df["brand_model_key"] = brand + "__" + model
        df["model_ref_key"] = model + "__" + ref_value

        return df


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = list(columns) if columns is not None else []

    def fit(self, x, y=None):
        self.frequency_maps_ = {}
        n_rows = len(x)
        if n_rows == 0:
            raise ValueError("Cannot fit FrequencyEncoder on an empty dataframe.")

        for col in self.columns:
            values = x[col].fillna("missing").astype(str)
            self.frequency_maps_[col] = (values.value_counts() / n_rows).to_dict()
        return self

    def transform(self, x):
        df = x.copy()
        for col in self.columns:
            values = df[col].fillna("missing").astype(str)
            df[f"{col}_frequency"] = values.map(self.frequency_maps_[col]).fillna(0.0)
        return df


class OOFTargetMeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        columns=None,
        smoothing=None,
        n_splits=5,
        random_state=RANDOM_STATE,
    ):
        self.columns = list(columns) if columns is not None else []
        self.smoothing = smoothing if smoothing is not None else {}
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, x, y):
        y_series = pd.Series(y, index=x.index, dtype=float)
        self.global_mean_ = float(y_series.mean())
        self.encoding_maps_ = {}

        for col in self.columns:
            self.encoding_maps_[col] = self._fit_column_mapping(x[col], y_series, col)

        return self

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)
        df = x.copy()
        y_series = pd.Series(y, index=df.index, dtype=float)

        n_splits = min(self.n_splits, len(df))
        if n_splits < 2:
            return self.transform(df)

        for col in self.columns:
            encoded_col = f"{col}_target_mean"
            df[encoded_col] = np.nan
            kf = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            )

            for train_idx, holdout_idx in kf.split(df):
                fold_x = df.iloc[train_idx]
                fold_y = y_series.iloc[train_idx]
                fold_mapping = self._fit_column_mapping(fold_x[col], fold_y, col)
                holdout_values = df.iloc[holdout_idx][col].fillna("missing").astype(str)
                df.iloc[
                    holdout_idx,
                    df.columns.get_loc(encoded_col),
                ] = holdout_values.map(fold_mapping).fillna(self.global_mean_).values

            df[encoded_col] = df[encoded_col].fillna(self.global_mean_)

        return df

    def transform(self, x):
        df = x.copy()
        for col in self.columns:
            values = df[col].fillna("missing").astype(str)
            df[f"{col}_target_mean"] = (
                values.map(self.encoding_maps_[col]).fillna(self.global_mean_)
            )
        return df

    def _fit_column_mapping(self, series, y, col):
        values = series.fillna("missing").astype(str)
        stats = (
            pd.DataFrame({"value": values, "target": y})
            .groupby("value")["target"]
            .agg(["mean", "count"])
        )
        smoothing = self._smoothing_for(col)
        smooth_mean = (
            stats["count"] * stats["mean"] + smoothing * self.global_mean_
        ) / (stats["count"] + smoothing)
        return smooth_mean.to_dict()

    def _smoothing_for(self, col):
        if isinstance(self.smoothing, dict):
            return self.smoothing.get(col, 20)
        return self.smoothing


def build_tree_feature_preprocessor(target_encoder_splits=5):
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )

    return Pipeline(
        [
            ("feature_builder", WatchTreeFeatureBuilder()),
            ("frequency_encoder", FrequencyEncoder(FREQUENCY_COLS)),
            (
                "target_encoder",
                OOFTargetMeanEncoder(
                    columns=OOF_TARGET_COLS,
                    smoothing=OOF_TARGET_SMOOTHING,
                    n_splits=target_encoder_splits,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "column_preprocessor",
                ColumnTransformer(
                    [
                        ("num", numeric_transformer, FINAL_NUMERIC_COLS),
                        ("cat", categorical_transformer, FINAL_CATEGORICAL_COLS),
                    ],
                    remainder="drop",
                ),
            ),
        ]
    )


def build_final_extra_tree_pipeline(
    n_estimators=TUNED_EXTRA_TREE_PARAMS["n_estimators"],
    n_jobs=TUNED_EXTRA_TREE_PARAMS["n_jobs"],
    target_encoder_splits=5,
    **extra_tree_overrides,
):
    params = TUNED_EXTRA_TREE_PARAMS.copy()
    params.update(extra_tree_overrides)
    params["n_estimators"] = n_estimators
    params["n_jobs"] = n_jobs

    return Pipeline(
        [
            ("preprocessor", build_tree_feature_preprocessor(target_encoder_splits)),
            ("model", ExtraTreesRegressor(**params)),
        ]
    )


def evaluate_model(model, x_valid, y_valid_log, y_valid_price):
    pred_log = model.predict(x_valid)
    pred_price = np.expm1(pred_log)
    y_valid_price = pd.Series(y_valid_price, dtype=float)
    nonzero_price = y_valid_price != 0
    mape_percent = (
        np.mean(
            np.abs(
                (y_valid_price.loc[nonzero_price] - pred_price[nonzero_price])
                / y_valid_price.loc[nonzero_price]
            )
        )
        * 100
    )
    return {
        "rmse_log": mean_squared_error(y_valid_log, pred_log) ** 0.5,
        "mae_dollars": mean_absolute_error(y_valid_price, pred_price),
        "rmse_dollars": mean_squared_error(y_valid_price, pred_price) ** 0.5,
        "r2_log": r2_score(y_valid_log, pred_log),
        "mape_percent": mape_percent,
    }


def evaluate_on_test(pipeline, test_df):
    x_test, y_test = split_features_target(test_df)
    metrics = evaluate_model(pipeline, x_test, y_test, test_df[TARGET_COL])

    pred_log = pipeline.predict(x_test)
    pred_price = np.expm1(pred_log)

    predictions = test_df.copy()
    predictions["pred_log"] = pred_log
    predictions["pred_price"] = pred_price
    predictions["signed_error"] = predictions[TARGET_COL] - predictions["pred_price"]
    predictions["abs_error"] = predictions["signed_error"].abs()
    predictions["abs_pct_error"] = predictions["abs_error"] / predictions[TARGET_COL]
    predictions["abs_log_error"] = (y_test - predictions["pred_log"]).abs()

    return metrics, predictions


def train_final_pipeline(train_df, **pipeline_kwargs):
    x_train, y_train = split_features_target(train_df)
    pipeline = build_final_extra_tree_pipeline(**pipeline_kwargs)
    pipeline.fit(x_train, y_train)
    return pipeline


def save_pipeline(pipeline, output_path=MODEL_OUTPUT_PATH):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    joblib.dump(pipeline, output_path)
    return output_path


def load_pipeline(path=MODEL_OUTPUT_PATH):
    return joblib.load(path)


def print_metrics(metrics):
    print("Test metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")


def run_modeling():
    train_df, test_df = load_cleaned_data()

    if os.path.exists(MODEL_OUTPUT_PATH):
        print(f"Loading pipeline from {MODEL_OUTPUT_PATH}", flush=True)
        pipeline = load_pipeline(MODEL_OUTPUT_PATH)
    else:
        print("Training tuned Extra Trees pipeline", flush=True)
        pipeline = train_final_pipeline(train_df)
        save_pipeline(pipeline, MODEL_OUTPUT_PATH)
        print(f"Saved pipeline to {MODEL_OUTPUT_PATH}", flush=True)

    metrics, predictions = evaluate_on_test(pipeline, test_df)
    print_metrics(metrics)
    return metrics, predictions


def main():
    return run_modeling()


if __name__ == "__main__":
    main()
