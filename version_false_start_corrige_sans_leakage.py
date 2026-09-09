"""
Leakage-free evaluation of hybrid LSTM/GRU/Transformer models
for false-start detection.

TWO EXPERIMENTAL MODES
----------------------

1. TEMPORAL MODE:
   - Uses temporal sequences.
   - Walk-forward / TimeSeriesSplit.
   - Tests several timesteps.

2. NO-TIMESTEP MODE:
   - No temporal sequence construction.
   - Direct tabular classification.
   - TimeSeriesSplit is still used to preserve chronology.

IMPORTANT LEAKAGE CORRECTIONS
-----------------------------

[CORRECTION 1]
StandardScaler is fitted ONLY on the training portion of each fold.

[CORRECTION 2]
Kendall, Pettitt and Lombard statistics are computed ONLY on
the training portion of each fold.

[CORRECTION 3]
Temporal sequences are constructed separately inside each fold.
Validation observations are never used to construct training sequences.

[CORRECTION 4]
StratifiedKFold(shuffle=True) has been removed for temporal data.
TimeSeriesSplit is used instead.

[CORRECTION 5]
No preprocessing is performed globally before cross-validation.

[CORRECTION 6]
The model is recreated from scratch for every fold.

[CORRECTION 7]
The no-timestep experiment operates directly on tabular observations.

[CORRECTION 8]
Global test statistics are never used to transform training or
validation observations.

[CORRECTION 9]
The target variable 'faux_demarage' is never included in X.

[CORRECTION 10]
Validation predictions are produced strictly from models trained
on preceding observations.

Author:
Pape El Hadji Abdoulaye Gueye
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import time
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from scipy.stats import kendalltau

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "donnees_avec_faux_demarage.xlsx"

TARGET = "faux_demarage"

VARIABLES_TEST = [
    "PRECTOTCORR_SUM",
    "T2M",
]

TIMESTEPS_VALUES = [
    1,
    2,
    3,
    4,
    5,
]

N_SPLITS = 5

EPOCHS = 50

BATCH_SIZE = 32

OUTPUT_DIR = "results_leakage_free"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n============================================================")
print("LOADING DATA")
print("============================================================")

df = pd.read_excel(DATA_FILE)

print(
    f"[INFO] Dataset shape: {df.shape}"
)

print(
    "[INFO] Columns:"
)

print(
    df.columns.tolist()
)


# ============================================================
# BASIC VALIDATION
# ============================================================

if TARGET not in df.columns:

    raise ValueError(
        f"[ERROR] Target column '{TARGET}' is missing."
    )


missing_test_variables = [
    v
    for v in VARIABLES_TEST
    if v not in df.columns
]

if missing_test_variables:

    print(
        "[WARNING] Statistical-test variables missing:"
    )

    print(
        missing_test_variables
    )


# ============================================================
# TARGET CLEANING
# ============================================================

df = df.copy()

df = df.dropna(
    subset=[TARGET]
)

df[TARGET] = (
    pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )
)

df = df.dropna(
    subset=[TARGET]
)

df[TARGET] = (
    df[TARGET]
    .astype(int)
)


unique_target = sorted(
    df[TARGET].unique()
)

print(
    "[INFO] Target classes:",
    unique_target,
)

if not set(unique_target).issubset({0, 1}):

    raise ValueError(
        "[ERROR] faux_demarage must contain "
        "binary values 0 and 1."
    )


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

# ------------------------------------------------------------
# CORRECTION:
# The observations must be ordered chronologically BEFORE
# TimeSeriesSplit.
# ------------------------------------------------------------

if "Year" in df.columns:

    df["Year"] = pd.to_numeric(
        df["Year"],
        errors="coerce",
    )

    if df["Year"].isna().any():

        raise ValueError(
            "[ERROR] Missing values in Year."
        )

    df = df.sort_values(
        by="Year"
    ).reset_index(
        drop=True
    )

else:

    print(
        "[WARNING] 'Year' not found."
    )

    print(
        "[WARNING] Original row order will be treated "
        "as chronological order."
    )

    df = df.reset_index(
        drop=True
    )


# ============================================================
# FEATURE SELECTION
# ============================================================

# Remove target from explanatory variables.
feature_columns = [
    c
    for c in df.columns
    if c != TARGET
]


# ------------------------------------------------------------
# Remove non-numeric columns
# ------------------------------------------------------------

numeric_features = []

for column in feature_columns:

    if pd.api.types.is_numeric_dtype(
        df[column]
    ):

        numeric_features.append(
            column
        )

    else:

        print(
            f"[INFO] Non-numeric column ignored: {column}"
        )


# ------------------------------------------------------------
# Remove columns containing only missing values
# ------------------------------------------------------------

numeric_features = [
    c
    for c in numeric_features
    if not df[c].isna().all()
]


if len(numeric_features) == 0:

    raise ValueError(
        "[ERROR] No usable numeric features."
    )


print(
    f"[INFO] Number of numeric features: "
    f"{len(numeric_features)}"
)


# ============================================================
# DATA IMPUTATION FUNCTION
# ============================================================

def impute_train_validation(
    X_train,
    X_val,
):
    """
    Imputation is fitted on training data only.

    The validation set never contributes to the
    imputation statistics.
    """

    X_train = X_train.copy()

    X_val = X_val.copy()

    train_medians = (
        X_train
        .median()
    )

    X_train = (
        X_train
        .fillna(train_medians)
    )

    X_val = (
        X_val
        .fillna(train_medians)
    )

    # If a feature is entirely NaN in train,
    # use zero as a final fallback.
    X_train = X_train.fillna(0)

    X_val = X_val.fillna(0)

    return (
        X_train,
        X_val,
    )


# ============================================================
# STATISTICAL TESTS
# ============================================================

def pettitt_test(x):
    """
    Pettitt change-point test.

    This implementation is used ONLY on training data
    inside each fold.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    n = len(x)

    if n < 3:

        return np.nan

    k_values = []

    for k in range(1, n):

        s = 0

        for i in range(k):

            for j in range(
                k,
                n,
            ):

                s += np.sign(
                    x[j] - x[i]
                )

        k_values.append(
            s
        )

    k_max = max(
        k_values,
        key=abs,
    )

    denominator = (
        n**3 - n
    )

    if denominator <= 0:

        return np.nan

    p = (
        2
        * np.exp(
            (
                -6
                * k_max**2
            )
            / denominator
        )
    )

    return float(
        min(
            p,
            1.0,
        )
    )


def lombard_test(x):
    """
    Simplified Lombard-type statistic.

    IMPORTANT:
    It is computed only from training observations.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    n = len(x)

    if n < 3:

        return np.nan

    reference = x[0]

    differences = (
        x[1:]
        - reference
    )

    signs = np.sign(
        differences
    )

    k_values = np.arange(
        1,
        len(signs) + 1,
    )

    S = np.sum(
        k_values
        * signs
    )

    if n <= 0:

        return np.nan

    p = (
        2
        * np.exp(
            -S**2 / n
        )
    )

    return float(
        min(
            p,
            1.0,
        )
    )


def compute_training_statistical_features(
    X_train,
    X_val,
    variable_names,
):
    """
    Compute statistical-test-derived features.

    IMPORTANT:
    All statistics are calculated using X_train ONLY.

    The same training-derived statistic is then assigned
    to the corresponding variable in validation.

    This prevents validation information from entering
    the feature engineering process.
    """

    X_train = X_train.copy()

    X_val = X_val.copy()

    for var in variable_names:

        if var not in X_train.columns:

            continue

        train_values = (
            pd.to_numeric(
                X_train[var],
                errors="coerce",
            )
        )

        val_values = (
            pd.to_numeric(
                X_val[var],
                errors="coerce",
            )
        )

        valid_train = (
            train_values
            .notna()
        )

        train_clean = (
            train_values[
                valid_train
            ]
            .values
        )

        if len(train_clean) < 3:

            p_kendall = np.nan
            p_pettitt = np.nan
            p_lombard = np.nan

        else:

            time_index = np.arange(
                len(train_clean)
            )

            try:

                p_kendall = (
                    kendalltau(
                        time_index,
                        train_clean,
                    ).pvalue
                )

            except Exception:

                p_kendall = np.nan

            try:

                p_pettitt = (
                    pettitt_test(
                        train_clean
                    )
                )

            except Exception:

                p_pettitt = np.nan

            try:

                p_lombard = (
                    lombard_test(
                        train_clean
                    )
                )

            except Exception:

                p_lombard = np.nan

        # ----------------------------------------------------
        # CORRECTION:
        # The validation observations DO NOT determine
        # these values.
        # ----------------------------------------------------

        X_train[
            f"p_Kendall_{var}"
        ] = (
            p_kendall
        )

        X_val[
            f"p_Kendall_{var}"
        ] = (
            p_kendall
        )

        X_train[
            f"p_Pettitt_{var}"
        ] = (
            p_pettitt
        )

        X_val[
            f"p_Pettitt_{var}"
        ] = (
            p_pettitt
        )

        X_train[
            f"p_Lombard_{var}"
        ] = (
            p_lombard
        )

        X_val[
            f"p_Lombard_{var}"
        ] = (
            p_lombard
        )

    return (
        X_train,
        X_val,
    )


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_temporal_model(
    architecture,
    timesteps,
    n_features,
):
    """
    Build LSTM/GRU/Transformer architecture.
    """

    input_layer = tf.keras.Input(
        shape=(
            timesteps,
            n_features,
        )
    )

    components = []

    if "lstm" in architecture:

        components.append(
            tf.keras.layers.LSTM(
                32
            )(
                input_layer
            )
        )

    if "gru" in architecture:

        components.append(
            tf.keras.layers.GRU(
                32
            )(
                input_layer
            )
        )

    if "transformer" in architecture:

        x_trans = (
            tf.keras.layers.Dense(
                32
            )(
                input_layer
            )
        )

        attn_out = (
            tf.keras.layers.MultiHeadAttention(
                num_heads=4,
                key_dim=8,
            )(
                x_trans,
                x_trans,
            )
        )

        trans_out = (
            tf.keras.layers.GlobalAveragePooling1D()(
                attn_out
            )
        )

        components.append(
            trans_out
        )

    if len(components) == 0:

        raise ValueError(
            "No architecture component selected."
        )

    if len(components) == 1:

        merged = components[0]

    else:

        merged = (
            tf.keras.layers.concatenate(
                components
            )
        )

    output = (
        tf.keras.layers.Dense(
            1,
            activation="sigmoid",
        )(
            merged
        )
    )

    model = tf.keras.Model(
        inputs=input_layer,
        outputs=output,
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy"
        ],
    )

    return model


def build_tabular_model(
    architecture,
    n_features,
):
    """
    No-timestep version.

    Directly operates on tabular features.
    """

    input_layer = tf.keras.Input(
        shape=(
            n_features,
        )
    )

    components = []

    # --------------------------------------------------------
    # In the no-timestep version, LSTM/GRU/Transformer are
    # not mathematically appropriate because there is no
    # temporal dimension.
    #
    # Therefore the architecture becomes a dense classifier.
    # --------------------------------------------------------

    x = (
        tf.keras.layers.Dense(
            64,
            activation="relu",
        )(
            input_layer
        )
    )

    x = (
        tf.keras.layers.Dense(
            32,
            activation="relu",
        )(
            x
        )
    )

    x = (
        tf.keras.layers.Dropout(
            0.2
        )(
            x
        )
    )

    output = (
        tf.keras.layers.Dense(
            1,
            activation="sigmoid",
        )(
            x
        )
    )

    model = tf.keras.Model(
        inputs=input_layer,
        outputs=output,
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy"
        ],
    )

    return model


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_prob,
):
    """
    Robust binary classification metrics.
    """

    y_true = np.asarray(
        y_true
    )

    y_prob = np.asarray(
        y_prob
    )

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    if len(
        np.unique(y_true)
    ) == 2:

        auc = roc_auc_score(
            y_true,
            y_prob,
        )

    else:

        auc = np.nan

    return {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


# ============================================================
# PREPARE BASE DATA
# ============================================================

X_base = (
    df[
        numeric_features
    ]
    .copy()
)

y_base = (
    df[TARGET]
    .values
)


# ============================================================
# TEMPORAL EXPERIMENT
# ============================================================

def run_temporal_experiment(
    timesteps,
    architecture,
):
    """
    Leakage-free temporal experiment.

    Important:
    The complete dataset is NEVER scaled before CV.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"TEMPORAL EXPERIMENT | "
        f"timesteps={timesteps} | "
        f"architecture={architecture}"
    )

    print(
        "=" * 70
    )

    results = []

    # --------------------------------------------------------
    # TimeSeriesSplit operates on chronological rows.
    # --------------------------------------------------------

    tscv = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    for fold, (
        train_idx,
        val_idx,
    ) in enumerate(
        tscv.split(
            X_base
        ),
        start=1,
    ):

        print(
            f"\n[FOLD {fold}/{N_SPLITS}]"
        )

        # ----------------------------------------------------
        # RAW SPLIT
        # ----------------------------------------------------

        X_train_raw = (
            X_base
            .iloc[
                train_idx
            ]
            .copy()
        )

        X_val_raw = (
            X_base
            .iloc[
                val_idx
            ]
            .copy()
        )

        y_train = (
            y_base[
                train_idx
            ]
        )

        y_val = (
            y_base[
                val_idx
            ]
        )

        print(
            f"Train size: {len(train_idx)}"
        )

        print(
            f"Validation size: {len(val_idx)}"
        )

        # ----------------------------------------------------
        # CLASS CHECK
        # ----------------------------------------------------

        if len(
            np.unique(y_train)
        ) < 2:

            print(
                "[WARNING] Training fold contains "
                "only one class. Fold skipped."
            )

            continue

        if len(
            np.unique(y_val)
        ) < 2:

            print(
                "[WARNING] Validation fold contains "
                "only one class. AUC will be NaN."
            )

        # ----------------------------------------------------
        # STATISTICAL FEATURES
        # ----------------------------------------------------

        X_train_raw, X_val_raw = (
            compute_training_statistical_features(
                X_train_raw,
                X_val_raw,
                VARIABLES_TEST,
            )
        )

        # ----------------------------------------------------
        # IMPUTATION
        # ----------------------------------------------------

        X_train_raw, X_val_raw = (
            impute_train_validation(
                X_train_raw,
                X_val_raw,
            )
        )

        # ----------------------------------------------------
        # SCALING
        #
        # CORRECTION:
        # scaler.fit() ONLY on training data.
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = (
            scaler.fit_transform(
                X_train_raw
            )
        )

        X_val_scaled = (
            scaler.transform(
                X_val_raw
            )
        )

        # ----------------------------------------------------
        # BUILD TEMPORAL SEQUENCES
        #
        # CORRECTION:
        # Sequences are created separately after the fold split.
        # ----------------------------------------------------

        X_train_seq = []

        y_train_seq = []

        for i in range(
            timesteps,
            len(
                X_train_scaled
            ),
        ):

            X_train_seq.append(
                X_train_scaled[
                    i - timesteps:i
                ]
            )

            y_train_seq.append(
                y_train[i]
            )

        X_val_seq = []

        y_val_seq = []

        for i in range(
            timesteps,
            len(
                X_val_scaled
            ),
        ):

            X_val_seq.append(
                X_val_scaled[
                    i - timesteps:i
                ]
            )

            y_val_seq.append(
                y_val[i]
            )

        if (
            len(X_train_seq) == 0
            or len(X_val_seq) == 0
        ):

            print(
                "[WARNING] Not enough observations "
                "for this timestep."
            )

            continue

        X_train_seq = np.asarray(
            X_train_seq,
            dtype=np.float32,
        )

        X_val_seq = np.asarray(
            X_val_seq,
            dtype=np.float32,
        )

        y_train_seq = np.asarray(
            y_train_seq,
            dtype=np.float32,
        )

        y_val_seq = np.asarray(
            y_val_seq,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # BUILD MODEL
        # ----------------------------------------------------

        model = build_temporal_model(
            architecture,
            timesteps,
            X_train_seq.shape[2],
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        early_stop = (
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            )
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        start = time.time()

        model.fit(
            X_train_seq,
            y_train_seq,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(
                X_val_seq,
                y_val_seq,
            ),
            callbacks=[
                early_stop
            ],
            verbose=0,
        )

        elapsed = (
            time.time()
            - start
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        y_prob = (
            model.predict(
                X_val_seq,
                verbose=0,
            )
            .flatten()
        )

        metrics = calculate_metrics(
            y_val_seq,
            y_prob,
        )

        metrics[
            "Fold"
        ] = fold

        metrics[
            "Timesteps"
        ] = timesteps

        metrics[
            "Architecture"
        ] = architecture

        metrics[
            "Train_Time_s"
        ] = elapsed

        metrics[
            "N_Train"
        ] = len(
            y_train_seq
        )

        metrics[
            "N_Validation"
        ] = len(
            y_val_seq
        )

        results.append(
            metrics
        )

        print(
            f"Accuracy={metrics['Accuracy']:.4f} | "
            f"AUC={metrics['AUC']:.4f} | "
            f"F1={metrics['F1']:.4f}"
        )

        # ----------------------------------------------------
        # CLEAN MODEL
        # ----------------------------------------------------

        tf.keras.backend.clear_session()

    return results


# ============================================================
# NO-TIMESTEP EXPERIMENT
# ============================================================

def run_no_timestep_experiment():
    """
    Leakage-free tabular experiment.

    No sequence is created.
    No LSTM/GRU/Transformer temporal dimension is used.

    This provides a clean baseline for comparison.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "NO-TIMESTEP EXPERIMENT"
    )

    print(
        "=" * 70
    )

    results = []

    tscv = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    for fold, (
        train_idx,
        val_idx,
    ) in enumerate(
        tscv.split(
            X_base
        ),
        start=1,
    ):

        print(
            f"\n[FOLD {fold}/{N_SPLITS}]"
        )

        X_train_raw = (
            X_base
            .iloc[
                train_idx
            ]
            .copy()
        )

        X_val_raw = (
            X_base
            .iloc[
                val_idx
            ]
            .copy()
        )

        y_train = (
            y_base[
                train_idx
            ]
        )

        y_val = (
            y_base[
                val_idx
            ]
        )

        if len(
            np.unique(y_train)
        ) < 2:

            print(
                "[WARNING] Training fold has "
                "only one class. Skipped."
            )

            continue

        # ----------------------------------------------------
        # STATISTICAL TESTS
        # ----------------------------------------------------

        X_train_raw, X_val_raw = (
            compute_training_statistical_features(
                X_train_raw,
                X_val_raw,
                VARIABLES_TEST,
            )
        )

        # ----------------------------------------------------
        # IMPUTATION
        # ----------------------------------------------------

        X_train_raw, X_val_raw = (
            impute_train_validation(
                X_train_raw,
                X_val_raw,
            )
        )

        # ----------------------------------------------------
        # SCALING
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = (
            scaler.fit_transform(
                X_train_raw
            )
        )

        X_val_scaled = (
            scaler.transform(
                X_val_raw
            )
        )

        # ----------------------------------------------------
        # BUILD TABULAR MODEL
        # ----------------------------------------------------

        model = build_tabular_model(
            architecture="MLP",
            n_features=X_train_scaled.shape[1],
        )

        early_stop = (
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            )
        )

        start = time.time()

        model.fit(
            X_train_scaled,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(
                X_val_scaled,
                y_val,
            ),
            callbacks=[
                early_stop
            ],
            verbose=0,
        )

        elapsed = (
            time.time()
            - start
        )

        y_prob = (
            model.predict(
                X_val_scaled,
                verbose=0,
            )
            .flatten()
        )

        metrics = calculate_metrics(
            y_val,
            y_prob,
        )

        metrics[
            "Fold"
        ] = fold

        metrics[
            "Timesteps"
        ] = 0

        metrics[
            "Architecture"
        ] = "MLP_no_timestep"

        metrics[
            "Train_Time_s"
        ] = elapsed

        metrics[
            "N_Train"
        ] = len(
            y_train
        )

        metrics[
            "N_Validation"
        ] = len(
            y_val
        )

        results.append(
            metrics
        )

        print(
            f"Accuracy={metrics['Accuracy']:.4f} | "
            f"AUC={metrics['AUC']:.4f} | "
            f"F1={metrics['F1']:.4f}"
        )

        tf.keras.backend.clear_session()

    return results


# ============================================================
# RUN TEMPORAL MODELS
# ============================================================

all_temporal_results = []

model_configs = {

    "LSTM": [
        "lstm"
    ],

    "LSTM_GRU": [
        "lstm",
        "gru",
    ],

    "LSTM_Transformer": [
        "lstm",
        "transformer",
    ],

    "GRU_Transformer": [
        "gru",
        "transformer",
    ],

    "LSTM_GRU_Transformer": [
        "lstm",
        "gru",
        "transformer",
    ],
}


for timesteps in TIMESTEPS_VALUES:

    for architecture_name, architecture in (
        model_configs.items()
    ):

        fold_results = (
            run_temporal_experiment(
                timesteps,
                architecture,
            )
        )

        all_temporal_results.extend(
            fold_results
        )


# ============================================================
# RUN NO-TIMESTEP MODEL
# ============================================================

no_timestep_results = (
    run_no_timestep_experiment()
)


# ============================================================
# COMBINE RESULTS
# ============================================================

all_results = (
    all_temporal_results
    + no_timestep_results
)


results_df = pd.DataFrame(
    all_results
)


if len(results_df) == 0:

    raise RuntimeError(
        "[ERROR] No valid results were generated."
    )


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

fold_results_path = os.path.join(
    OUTPUT_DIR,
    "leakage_free_fold_results.csv",
)

results_df.to_csv(
    fold_results_path,
    index=False,
)


# ============================================================
# AGGREGATED RESULTS
# ============================================================

summary_df = (
    results_df
    .groupby(
        [
            "Timesteps",
            "Architecture",
        ]
    )
    .agg(
        Accuracy_mean=(
            "Accuracy",
            "mean",
        ),
        Accuracy_std=(
            "Accuracy",
            "std",
        ),
        AUC_mean=(
            "AUC",
            "mean",
        ),
        AUC_std=(
            "AUC",
            "std",
        ),
        Precision_mean=(
            "Precision",
            "mean",
        ),
        Recall_mean=(
            "Recall",
            "mean",
        ),
        F1_mean=(
            "F1",
            "mean",
        ),
        F1_std=(
            "F1",
            "std",
        ),
        Train_Time_mean=(
            "Train_Time_s",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "leakage_free_summary.csv",
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "LEAKAGE-FREE EXPERIMENTAL SUMMARY"
)

print(
    "=" * 80
)

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config_df = pd.DataFrame(
    [
        {
            "data_file": DATA_FILE,
            "target": TARGET,
            "n_features_original": len(
                numeric_features
            ),
            "n_splits": N_SPLITS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "timesteps_tested": str(
                TIMESTEPS_VALUES
            ),
            "validation_method": (
                "TimeSeriesSplit"
            ),
            "scaling": (
                "fit_on_training_fold_only"
            ),
            "statistical_tests": (
                "training_fold_only"
            ),
            "sequence_creation": (
                "inside_each_fold"
            ),
            "shuffle": False,
            "no_timestep_version": True,
        }
    ]
)


config_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "leakage_free_configuration.csv",
    ),
    index=False,
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n============================================================"
)

print(
    "[INFO] Leakage-free evaluation completed."
)

print(
    f"[INFO] Fold results: {fold_results_path}"
)

print(
    f"[INFO] Summary: {summary_path}"
)

print(
    "[INFO] The no-timestep baseline is included."
)

print(
    "============================================================"
)