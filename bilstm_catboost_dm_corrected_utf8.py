"""
BiLSTM vs CatBoost + Diebold-Mariano test.
Leakage-free walk-forward evaluation.
Compatible with databasecor_augmented.xlsx.
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import time
import psutil

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Bidirectional,
    Concatenate,
    GaussianNoise,
)
from tensorflow.keras.callbacks import EarlyStopping
from catboost import CatBoostRegressor


# ============================================================
# CONFIGURATION
# ============================================================

DROP_SAVI_IF_CONSTANT = True
CLIP_NEGATIVE_SOIL_VALUES = True
SAVI_STD_THRESHOLD = 1e-6
MIN_REAL_POINTS_FOR_FOLD_METRICS = 2

file_path = "pcy_annexe/databasecor_augmented.xlsx"

df = pd.read_excel(file_path)

print("[INFO] Detected columns:")
print(df.columns.tolist())


# ============================================================
# GROWTH DURATION
# ============================================================

if "Growth_Duration" in df.columns:

    print("[INFO] Growth_Duration already exists.")

elif (
    "Growth_Start" in df.columns
    and "Growth_End" in df.columns
):

    df["Growth_Start"] = pd.to_datetime(
        df["Growth_Start"],
        errors="coerce",
    )

    df["Growth_End"] = pd.to_datetime(
        df["Growth_End"],
        errors="coerce",
    )

    df["Growth_Duration"] = (
        df["Growth_End"] - df["Growth_Start"]
    ).dt.days

    print("[INFO] Growth_Duration calculated.")

else:

    raise ValueError(
        "[ERROR] Growth_Duration or Growth_Start/Growth_End "
        "are required."
    )


# ============================================================
# YEAR COLUMN
# ============================================================

if "Year" in df.columns:

    year_col = "Year"

elif "Growth_Start" in df.columns:

    df["__Year"] = df["Growth_Start"].dt.year
    year_col = "__Year"

else:

    raise ValueError(
        "[ERROR] No usable year column found."
    )


if df[year_col].isna().any():

    raise ValueError(
        "[ERROR] Missing years detected."
    )


# ============================================================
# SYNTHETIC OBSERVATIONS
# ============================================================

if "is_synthetic" not in df.columns:

    print(
        "[WARNING] is_synthetic not found. "
        "All observations are treated as real."
    )

    df["is_synthetic"] = False


# Ensure boolean type
df["is_synthetic"] = (
    df["is_synthetic"]
    .fillna(False)
    .astype(bool)
)


# ============================================================
# FEATURES
# ============================================================

dynamic_features = [
    "Precipitation (mm)",
    "Temperature Min (°C)",
    "Temperature Max (°C)",
]


static_features_candidates = [
    "NDVI",
    "EVI",
    "SAVI",
    "Growth_Duration",
    "Area Harvested (ha)",
    "nitrogene",
    "phosphate",
    "Potassium",
    "Agricultural land(ha)",
    "Previous_Yield",
]


# ============================================================
# CHECK REQUIRED DYNAMIC FEATURES
# ============================================================

missing_dynamic = [
    c
    for c in dynamic_features
    if c not in df.columns
]

if missing_dynamic:

    raise ValueError(
        "[ERROR] Missing dynamic columns: "
        f"{missing_dynamic}"
    )


# ============================================================
# STATIC FEATURES
# ============================================================

missing_static = [
    c
    for c in static_features_candidates
    if c not in df.columns
    and c != "Previous_Yield"
]

if missing_static:

    print(
        "[INFO] Missing static columns ignored:",
        missing_static,
    )


static_features = [
    c
    for c in static_features_candidates
    if c in df.columns
    or c == "Previous_Yield"
]


target_feature = "Yield"

if target_feature not in df.columns:

    raise ValueError(
        f"[ERROR] Target column '{target_feature}' "
        "not found."
    )


# ============================================================
# MODEL PARAMETERS
# ============================================================

time_steps = 3
n_splits = 5
batch_size = 8
epochs = 100


# ============================================================
# REMOVE CONSTANT SAVI
# ============================================================

if (
    DROP_SAVI_IF_CONSTANT
    and "SAVI" in df.columns
    and df["SAVI"].std() < SAVI_STD_THRESHOLD
):

    print(
        "[INFO] SAVI is almost constant "
        "and will be removed."
    )

    if "SAVI" in static_features:

        static_features.remove("SAVI")


elif "SAVI" not in df.columns:

    print("[INFO] SAVI is not present.")


# ============================================================
# CLIP NEGATIVE SOIL VALUES
# ============================================================

if CLIP_NEGATIVE_SOIL_VALUES:

    for col in [
        "nitrogene",
        "phosphate",
        "Potassium",
    ]:

        if col in df.columns:

            n_neg = int(
                (df[col] < 0).sum()
            )

            if n_neg > 0:

                print(
                    f"[INFO] {col}: "
                    f"{n_neg} negative values clipped."
                )

                df[col] = df[col].clip(
                    lower=0
                )


print(
    "[INFO] Dynamic features:",
    dynamic_features,
)

print(
    "[INFO] Static features:",
    static_features,
)


# ============================================================
# GROUP COLUMN
# ============================================================

def infer_or_build_group_column(data):

    candidates = [
        "Group",
        "group",
        "Site",
        "site",
        "Location",
        "location",
        "Field_ID",
        "FieldId",
        "Plot",
        "plot",
        "Farm",
        "farm",
        "Station",
        "station",
        "ID",
        "id",
    ]

    for column in candidates:

        if column in data.columns:

            print(
                f"[INFO] Group column detected: "
                f"{column}"
            )

            return column

    data["__global_group"] = "ALL"

    print(
        "[WARNING] No group column found. "
        "One global time series will be used."
    )

    return "__global_group"


group_col = infer_or_build_group_column(df)


# ============================================================
# CREATE TEMPORAL SEQUENCES
# ============================================================

def create_sequences(
    data,
    dynamic_cols,
    static_cols,
    target_col,
    steps,
    group_col,
    sort_col,
):

    X_dyn = []
    X_stat = []
    y = []
    idxs = []

    for _, group_data in data.groupby(
        group_col,
        sort=False,
    ):

        if sort_col in group_data.columns:

            group_data = group_data.sort_values(
                by=sort_col
            )

        else:

            group_data = group_data.sort_index()

        if len(group_data) <= steps:
            continue

        for i in range(
            len(group_data) - steps
        ):

            X_dyn.append(
                group_data[
                    dynamic_cols
                ].iloc[
                    i:i + steps
                ].values
            )

            X_stat.append(
                group_data[
                    static_cols
                ].iloc[
                    i + steps
                ].values
            )

            y.append(
                group_data[
                    target_col
                ].iloc[
                    i + steps
                ]
            )

            idxs.append(
                group_data.index[
                    i + steps
                ]
            )

    return (
        np.asarray(X_dyn),
        np.asarray(X_stat),
        np.asarray(y),
        np.asarray(idxs),
    )


# ============================================================
# UNIQUE YEARS
# ============================================================

unique_years = np.sort(
    df[year_col].unique()
)


if len(unique_years) < n_splits + 1:

    n_splits = max(
        2,
        len(unique_years) - 1,
    )

    print(
        f"[WARNING] n_splits adjusted "
        f"to {n_splits}."
    )


if n_splits < 2:

    raise ValueError(
        "[ERROR] Not enough unique years "
        "for TimeSeriesSplit."
    )


tscv = TimeSeriesSplit(
    n_splits=n_splits
)


# ============================================================
# STORAGE
# ============================================================

bilstm_rows = []
catboost_rows = []
fold_year_ranges = []

all_y_true_real = []
all_pred_bilstm_real = []
all_pred_catboost_real = []


print(
    f"[INFO] Starting walk-forward CV "
    f"with {n_splits} folds."
)


# ============================================================
# WALK-FORWARD CROSS-VALIDATION
# ============================================================

for fold, (
    train_year_idx,
    val_year_idx,
) in enumerate(
    tscv.split(unique_years)
):

    print(
        f"\n========== "
        f"FOLD {fold + 1}/{n_splits} "
        f"=========="
    )


    # --------------------------------------------------------
    # YEARS
    # --------------------------------------------------------

    train_years = unique_years[
        train_year_idx
    ]

    val_years = unique_years[
        val_year_idx
    ]


    # --------------------------------------------------------
    # LEAKAGE CHECK
    # --------------------------------------------------------

    assert (
        train_years.max()
        < val_years.min()
    ), (
        f"[LEAKAGE] Fold {fold + 1}: "
        "train and validation years overlap."
    )


    fold_year_ranges.append(
        {
            "fold": fold + 1,
            "train_year_min": float(
                train_years.min()
            ),
            "train_year_max": float(
                train_years.max()
            ),
            "val_year_min": float(
                val_years.min()
            ),
            "val_year_max": float(
                val_years.max()
            ),
        }
    )


    print(
        f"[CHECK] Train "
        f"{train_years.min():.2f}-"
        f"{train_years.max():.2f} "
        f"< Val "
        f"{val_years.min():.2f}-"
        f"{val_years.max():.2f}"
    )


    # --------------------------------------------------------
    # TRAIN / VALIDATION DATA
    # --------------------------------------------------------

    df_train = df[
        df[year_col].isin(train_years)
    ].copy()

    df_val = df[
        df[year_col].isin(val_years)
    ].copy()


    n_real_val_rows = int(
        (~df_val["is_synthetic"]).sum()
    )


    print(
        f"[INFO] Validation rows: "
        f"{len(df_val)}, "
        f"real: {n_real_val_rows}"
    )


    # --------------------------------------------------------
    # SORT BEFORE PREVIOUS_YIELD
    # --------------------------------------------------------

    df_train = df_train.sort_values(
        by=[group_col, year_col]
    )

    df_val = df_val.sort_values(
        by=[group_col, year_col]
    )


    # --------------------------------------------------------
    # PREVIOUS YIELD
    # --------------------------------------------------------

    df_train["Previous_Yield"] = (
        df_train
        .groupby(group_col)[target_feature]
        .shift(1)
    )


    train_prev_mean = (
        df_train["Previous_Yield"]
        .mean()
    )


    if pd.isna(train_prev_mean):

        train_prev_mean = (
            df_train[target_feature]
            .mean()
        )


    df_train["Previous_Yield"] = (
        df_train["Previous_Yield"]
        .fillna(train_prev_mean)
    )


    # --------------------------------------------------------
    # VALIDATION PREVIOUS YIELD
    # --------------------------------------------------------

    df_val["Previous_Yield"] = (
        df_val
        .groupby(group_col)[target_feature]
        .shift(1)
    )


    df_val["Previous_Yield"] = (
        df_val["Previous_Yield"]
        .fillna(train_prev_mean)
    )


    # --------------------------------------------------------
    # GROWTH DURATION
    # --------------------------------------------------------

    train_growth_mean = (
        df_train["Growth_Duration"]
        .mean()
    )


    df_train["Growth_Duration"] = (
        df_train["Growth_Duration"]
        .fillna(train_growth_mean)
    )


    df_val["Growth_Duration"] = (
        df_val["Growth_Duration"]
        .fillna(train_growth_mean)
    )


    # --------------------------------------------------------
    # SCALERS
    # --------------------------------------------------------

    scaler_dyn = MinMaxScaler()
    scaler_stat = MinMaxScaler()
    scaler_target = MinMaxScaler()


    # --------------------------------------------------------
    # FIT ONLY ON TRAINING DATA
    # --------------------------------------------------------

    df_train[dynamic_features] = (
        scaler_dyn.fit_transform(
            df_train[dynamic_features]
        )
    )


    df_train[static_features] = (
        scaler_stat.fit_transform(
            df_train[static_features]
        )
    )


    df_train[[target_feature]] = (
        scaler_target.fit_transform(
            df_train[[target_feature]]
        )
    )


    # --------------------------------------------------------
    # TRANSFORM VALIDATION DATA
    # --------------------------------------------------------

    df_val[dynamic_features] = (
        scaler_dyn.transform(
            df_val[dynamic_features]
        )
    )


    df_val[static_features] = (
        scaler_stat.transform(
            df_val[static_features]
        )
    )


    df_val[[target_feature]] = (
        scaler_target.transform(
            df_val[[target_feature]]
        )
    )


    # --------------------------------------------------------
    # SEQUENCES
    # --------------------------------------------------------

    (
        X_dyn_train,
        X_stat_train,
        y_train,
        _,
    ) = create_sequences(
        df_train,
        dynamic_features,
        static_features,
        target_feature,
        time_steps,
        group_col,
        sort_col=year_col,
    )


    (
        X_dyn_val,
        X_stat_val,
        y_val,
        idxs_val,
    ) = create_sequences(
        df_val,
        dynamic_features,
        static_features,
        target_feature,
        time_steps,
        group_col,
        sort_col=year_col,
    )


    if (
        len(X_dyn_train) == 0
        or len(X_dyn_val) == 0
    ):

        print(
            f"[WARNING] Fold {fold + 1} skipped."
        )

        continue


    # --------------------------------------------------------
    # REAL VALIDATION MASK
    # --------------------------------------------------------

    real_mask_val = (
        ~df.loc[
            idxs_val,
            "is_synthetic",
        ].values
    )


    # ========================================================
    # BiLSTM MODEL
    # ========================================================

    dynamic_input = Input(
        shape=(
            time_steps,
            len(dynamic_features),
        ),
        name="dynamic_input",
    )


    x = GaussianNoise(
        0.1
    )(dynamic_input)


    x = Bidirectional(
        LSTM(
            280,
            return_sequences=True,
        )
    )(x)


    x = Bidirectional(
        LSTM(
            280,
            return_sequences=True,
        )
    )(x)


    x = Bidirectional(
        LSTM(
            280,
            return_sequences=False,
        )
    )(x)


    # --------------------------------------------------------
    # STATIC INPUT
    # --------------------------------------------------------

    static_input = Input(
        shape=(
            len(static_features),
        ),
        name="static_input",
    )


    yb = Dense(
        100,
        activation="relu",
    )(static_input)


    yb = Dense(
        100,
        activation="relu",
    )(yb)


    # --------------------------------------------------------
    # CONCATENATION
    # --------------------------------------------------------

    concat = Concatenate()(
        [x, yb]
    )


    # --------------------------------------------------------
    # DENSE HEAD
    # --------------------------------------------------------

    z = Dense(
        100,
        activation="relu",
    )(concat)


    z = Dense(
        100,
        activation="relu",
    )(z)


    z = Dense(
        100,
        activation="relu",
    )(z)


    z = Dense(
        100,
        activation="relu",
    )(z)


    output = Dense(
        1,
        activation="linear",
    )(z)


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    bilstm_model = Model(
        inputs=[
            dynamic_input,
            static_input,
        ],
        outputs=output,
    )


    bilstm_model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )


    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=0,
    )


    # ========================================================
    # BiLSTM TRAINING
    # ========================================================

    start_time = time.time()


    start_mem = (
        psutil.Process()
        .memory_info()
        .rss
        / 1024**2
    )


    bilstm_model.fit(
        [
            X_dyn_train,
            X_stat_train,
        ],
        y_train,
        validation_data=(
            [
                X_dyn_val,
                X_stat_val,
            ],
            y_val,
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )


    elapsed_bilstm = (
        time.time() - start_time
    )


    mem_bilstm = (
        psutil.Process()
        .memory_info()
        .rss
        / 1024**2
        - start_mem
    )


    # ========================================================
    # BiLSTM PREDICTION
    # ========================================================

    pred_bilstm_scaled = (
        bilstm_model.predict(
            [
                X_dyn_val,
                X_stat_val,
            ],
            verbose=0,
        )
    )


    pred_bilstm = (
        scaler_target
        .inverse_transform(
            pred_bilstm_scaled
        )
        .flatten()
    )


    y_val_actual = (
        scaler_target
        .inverse_transform(
            y_val.reshape(-1, 1)
        )
        .flatten()
    )


    # ========================================================
    # CATBOOST INPUT
    # ========================================================

    X_cat_train = np.hstack(
        [
            X_dyn_train.reshape(
                X_dyn_train.shape[0],
                -1,
            ),
            X_stat_train,
        ]
    )


    X_cat_val = np.hstack(
        [
            X_dyn_val.reshape(
                X_dyn_val.shape[0],
                -1,
            ),
            X_stat_val,
        ]
    )


    y_cat_train = y_train.ravel()


    # ========================================================
    # CATBOOST MODEL
    # ========================================================

    cat_model = CatBoostRegressor(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        verbose=0,
        random_state=42,
    )


    # ========================================================
    # CATBOOST TRAINING
    # ========================================================

    start_time_cb = time.time()


    start_mem_cb = (
        psutil.Process()
        .memory_info()
        .rss
        / 1024**2
    )


    cat_model.fit(
        X_cat_train,
        y_cat_train,
    )


    elapsed_cb = (
        time.time()
        - start_time_cb
    )


    mem_cb = (
        psutil.Process()
        .memory_info()
        .rss
        / 1024**2
        - start_mem_cb
    )


    # ========================================================
    # CATBOOST PREDICTION
    # ========================================================

    pred_cat_scaled = (
        cat_model.predict(
            X_cat_val
        )
    )


    pred_cat = (
        scaler_target
        .inverse_transform(
            pred_cat_scaled.reshape(
                -1,
                1,
            )
        )
        .flatten()
    )


    # ========================================================
    # DIAGNOSTIC METRICS
    # ========================================================

    rmse_b_all = np.sqrt(
        mean_squared_error(
            y_val_actual,
            pred_bilstm,
        )
    )


    rmse_c_all = np.sqrt(
        mean_squared_error(
            y_val_actual,
            pred_cat,
        )
    )


    print(
        f"[DIAGNOSTIC] "
        f"BiLSTM RMSE={rmse_b_all:.2f} | "
        f"CatBoost RMSE={rmse_c_all:.2f}"
    )


    # ========================================================
    # REAL OBSERVATIONS ONLY
    # ========================================================

    n_real = int(
        real_mask_val.sum()
    )


    if (
        n_real
        >= MIN_REAL_POINTS_FOR_FOLD_METRICS
    ):

        y_real = (
            y_val_actual[
                real_mask_val
            ]
        )


        pred_b_real = (
            pred_bilstm[
                real_mask_val
            ]
        )


        pred_c_real = (
            pred_cat[
                real_mask_val
            ]
        )


        # ----------------------------------------------------
        # BiLSTM METRICS
        # ----------------------------------------------------

        rmse_b = np.sqrt(
            mean_squared_error(
                y_real,
                pred_b_real,
            )
        )


        mape_b = (
            mean_absolute_percentage_error(
                y_real,
                pred_b_real,
            )
        )


        r2_b = (
            r2_score(
                y_real,
                pred_b_real,
            )
            if len(
                np.unique(y_real)
            ) > 1
            else np.nan
        )


        # ----------------------------------------------------
        # CatBoost METRICS
        # ----------------------------------------------------

        rmse_c = np.sqrt(
            mean_squared_error(
                y_real,
                pred_c_real,
            )
        )


        mape_c = (
            mean_absolute_percentage_error(
                y_real,
                pred_c_real,
            )
        )


        r2_c = (
            r2_score(
                y_real,
                pred_c_real,
            )
            if len(
                np.unique(y_real)
            ) > 1
            else np.nan
        )


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print(
            f"[BiLSTM REAL n={n_real}] "
            f"Fold {fold + 1} -> "
            f"RMSE={rmse_b:.2f} | "
            f"MAPE={mape_b * 100:.2f}% | "
            f"R2={r2_b:.4f} | "
            f"Time={elapsed_bilstm:.1f}s | "
            f"Mem={mem_bilstm:.1f}MB"
        )


        print(
            f"[CatBoost REAL n={n_real}] "
            f"Fold {fold + 1} -> "
            f"RMSE={rmse_c:.2f} | "
            f"MAPE={mape_c * 100:.2f}% | "
            f"R2={r2_c:.4f} | "
            f"Time={elapsed_cb:.1f}s | "
            f"Mem={mem_cb:.1f}MB"
        )


        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        bilstm_rows.append(
            {
                "fold": fold + 1,
                "n_real_val": n_real,
                "rmse": rmse_b,
                "mape_percent": (
                    mape_b * 100
                ),
                "r2": r2_b,
                "train_time_s": (
                    elapsed_bilstm
                ),
                "mem_delta_mb": (
                    mem_bilstm
                ),
            }
        )


        catboost_rows.append(
            {
                "fold": fold + 1,
                "n_real_val": n_real,
                "rmse": rmse_c,
                "mape_percent": (
                    mape_c * 100
                ),
                "r2": r2_c,
                "train_time_s": (
                    elapsed_cb
                ),
                "mem_delta_mb": (
                    mem_cb
                ),
            }
        )


        # ----------------------------------------------------
        # STORE REAL BACKTEST PREDICTIONS
        # ----------------------------------------------------

        all_y_true_real.append(
            y_real
        )


        all_pred_bilstm_real.append(
            pred_b_real
        )


        all_pred_catboost_real.append(
            pred_c_real
        )


    else:

        print(
            f"[WARNING] Fold {fold + 1}: "
            f"only {n_real} real observations."
        )


# ============================================================
# DIEBOLD-MARIANO TEST
# ============================================================

def diebold_mariano(
    y_true,
    y_pred1,
    y_pred2,
):

    e1 = (
        y_true - y_pred1
    )

    e2 = (
        y_true - y_pred2
    )

    # Squared-error loss differential
    d = (
        e1**2 - e2**2
    )

    mean_d = np.mean(d)

    var_d = np.var(
        d,
        ddof=1,
    )

    if (
        len(d) < 2
        or var_d <= 0
    ):

        return np.nan

    dm_stat = (
        mean_d
        / np.sqrt(
            var_d / len(d)
        )
    )

    return dm_stat


# ============================================================
# GLOBAL BACKTEST
# ============================================================

if all_y_true_real:

    y_true_full = np.concatenate(
        all_y_true_real
    )


    pred_bilstm_full = (
        np.concatenate(
            all_pred_bilstm_real
        )
    )


    pred_cat_full = (
        np.concatenate(
            all_pred_catboost_real
        )
    )


    print(
        "[INFO] Real backtest observations: "
        f"{len(y_true_full)}"
    )


    if (
        len(y_true_full)
        >= MIN_REAL_POINTS_FOR_FOLD_METRICS
    ):

        dm_stat = diebold_mariano(
            y_true_full,
            pred_bilstm_full,
            pred_cat_full,
        )


        print(
            "\n=============================="
        )


        print(
            "DIEBOLD-MARIANO TEST"
        )


        print(
            f"DM statistic: "
            f"{dm_stat:.4f}"
        )


        print(
            "|DM| > 1.96 indicates "
            "significance at 5%."
        )


        print(
            "Positive DM favors CatBoost "
            "under the defined loss."
        )


        print(
            "Negative DM favors BiLSTM "
            "under the defined loss."
        )


        print(
            "=============================="
        )


        # ----------------------------------------------------
        # GLOBAL RMSE
        # ----------------------------------------------------

        rmse_global_b = np.sqrt(
            mean_squared_error(
                y_true_full,
                pred_bilstm_full,
            )
        )


        rmse_global_c = np.sqrt(
            mean_squared_error(
                y_true_full,
                pred_cat_full,
            )
        )


        # ----------------------------------------------------
        # GLOBAL R2
        # ----------------------------------------------------

        r2_global_b = (
            r2_score(
                y_true_full,
                pred_bilstm_full,
            )
            if len(
                np.unique(y_true_full)
            ) > 1
            else np.nan
        )


        r2_global_c = (
            r2_score(
                y_true_full,
                pred_cat_full,
            )
            if len(
                np.unique(y_true_full)
            ) > 1
            else np.nan
        )


        print(
            f"BiLSTM: "
            f"RMSE={rmse_global_b:.2f} "
            f"| R2={r2_global_b:.4f}"
        )


        print(
            f"CatBoost: "
            f"RMSE={rmse_global_c:.2f} "
            f"| R2={r2_global_c:.4f}"
        )


        # ====================================================
        # SAVE DM RESULT
        # ====================================================

        pd.DataFrame(
            [
                {
                    "dm_statistic": dm_stat,
                    "rmse_global_bilstm": (
                        rmse_global_b
                    ),
                    "r2_global_bilstm": (
                        r2_global_b
                    ),
                    "rmse_global_catboost": (
                        rmse_global_c
                    ),
                    "r2_global_catboost": (
                        r2_global_c
                    ),
                    "n_obs_total_real": (
                        len(y_true_full)
                    ),
                }
            ]
        ).to_csv(
            "dm_test_global_result.csv",
            index=False,
        )


        # ====================================================
        # SAVE BACKTEST PREDICTIONS
        # ====================================================

        pd.DataFrame(
            {
                "y_true": y_true_full,
                "pred_bilstm": (
                    pred_bilstm_full
                ),
                "pred_catboost": (
                    pred_cat_full
                ),
            }
        ).to_csv(
            "backtest_predictions_concatenated.csv",
            index=False,
        )


    else:

        print(
            "[WARNING] Not enough real observations "
            "for the DM test."
        )


else:

    print(
        "[WARNING] No real validation observations "
        "available for the DM test."
    )


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

pd.DataFrame(
    bilstm_rows
).to_csv(
    "bilstm_cv_summary.csv",
    index=False,
)


pd.DataFrame(
    catboost_rows
).to_csv(
    "catboost_cv_summary.csv",
    index=False,
)


pd.DataFrame(
    fold_year_ranges
).to_csv(
    "temporal_fold_year_ranges.csv",
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "[INFO] Fold results saved."
)


print(
    pd.DataFrame(
        fold_year_ranges
    ).to_string(
        index=False
    )
)


print(
    "[INFO] Completed."
)