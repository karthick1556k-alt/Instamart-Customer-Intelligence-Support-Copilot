from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, silhouette_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import clone

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")

RFM_COLS = ["recency", "frequency", "monetary"]

NUMERIC = [
    "recency", "frequency", "monetary", "avg_discount",
    "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"
]
CATEGORICAL = ["favourite_category", "city", "gender"]
FEATURES = NUMERIC + CATEGORICAL


def load_data():
    customers = pd.read_csv(DATA / "customers.csv")
    orders = pd.read_csv(DATA / "orders.csv")
    products = pd.read_csv(DATA / "products.csv")

    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="raise")
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="raise"
    )

    orders["status"] = orders["status"].astype(str).str.lower()

    orders = orders.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    return customers, orders


def build_rfm(customers, orders):
    completed = orders[orders["status"].eq("completed")].copy()
    completed = completed[completed["order_ts"] < SNAPSHOT]

    rfm = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()

    rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
    rfm = rfm[["customer_id"] + RFM_COLS].copy()

    print("\n========== RFM VALIDATION ==========")
    print("RFM customers:", len(rfm))
    print("Recency median:", rfm["recency"].median())
    print("Recency minimum:", rfm["recency"].min())
    print("Recency maximum:", rfm["recency"].max())
    print("Frequency mean:", round(rfm["frequency"].mean(), 2))
    print("Frequency maximum:", rfm["frequency"].max())
    print("Monetary mean:", round(rfm["monetary"].mean(), 2))
    print("Monetary maximum:", round(rfm["monetary"].max(), 2))

    expected = {
        "customer_count": len(rfm) == 1175,
        "recency_median": rfm["recency"].median() == 19,
        "recency_min": rfm["recency"].min() == 0,
        "recency_max": rfm["recency"].max() == 180,
        "frequency_max": rfm["frequency"].max() == 33,
        "monetary_max": np.isclose(rfm["monetary"].max(), 14766.55, atol=0.01)
    }

    for name, passed in expected.items():
        print(f"{'PASS' if passed else 'CHECK'}: {name}")

    if len(rfm) != 1175:
        raise AssertionError(
            f"Expected 1175 RFM customers, found {len(rfm)}. Stop and investigate."
        )

    return rfm


def cluster_customers(rfm):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[RFM_COLS])

    print("\n========== K-MEANS EVALUATION ==========")
    rows = []

    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(scaled)
        score = silhouette_score(scaled, labels)

        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": score
        })

        print(
            f"K={k}: inertia={km.inertia_:.1f}, "
            f"silhouette={score:.3f}"
        )

    pd.DataFrame(rows).to_csv(
        OUT / "part_b_kmeans_evaluation.csv", index=False
    )

    # K=4 is selected based on the elbow, the reference silhouette
    # score, and the business value of four actionable segments.
    final_k = 4
    model = KMeans(n_clusters=final_k, n_init=10, random_state=42)
    rfm["segment_id"] = model.fit_predict(scaled)

    profile = rfm.groupby("segment_id").agg(
        customer_count=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    ).reset_index()

    premium_id = int(
        profile.loc[profile["mean_monetary"].idxmax(), "segment_id"]
    )
    frequency_id = int(
        profile.loc[profile["mean_frequency"].idxmax(), "segment_id"]
    )
    at_risk_id = int(
        profile.loc[profile["mean_recency"].idxmax(), "segment_id"]
    )

    names = {}
    for sid in profile["segment_id"]:
        sid = int(sid)
        if sid == premium_id:
            names[sid] = "Premium"
        elif sid == frequency_id:
            names[sid] = "Frequent Shoppers"
        elif sid == at_risk_id:
            names[sid] = "At-Risk Customers"
        else:
            names[sid] = "Growth Customers"

    # Guarantee unique names if two profile rankings point to one cluster.
    used = set()
    for sid in sorted(names):
        if names[sid] in used:
            names[sid] = f"Growth Segment {sid}"
        used.add(names[sid])

    rfm["segment_name"] = rfm["segment_id"].map(names)
    rfm["is_premium"] = rfm["segment_id"].eq(premium_id)

    print("\n========== SEGMENT PROFILE ==========")
    print(profile.round(2).to_string(index=False))
    print("\nSegment names and recommended actions:")
    actions = {
        "Premium": "VIP rewards, early access and personalised retention.",
        "Frequent Shoppers": "Loyalty points and repeat-purchase incentives.",
        "At-Risk Customers": "Win-back offers and re-engagement campaigns.",
        "Growth Customers": "Cross-sell recommendations and basket-building offers."
    }

    for sid, name in names.items():
        print(f"Cluster {sid}: {name} — {actions.get(name, 'Targeted engagement.')}")
    print("Premium cluster ID:", premium_id)
    print("Note: cluster numbers are arbitrary labels, not an ordered ranking.")

    profile.to_csv(OUT / "part_b_segment_profile.csv", index=False)

    pca = PCA(n_components=2)
    pca.fit(scaled)
    ratios = pca.explained_variance_ratio_

    print("\nPCA explained variance:", ratios.round(3))
    print("PCA cumulative variance:", round(ratios.sum(), 3))

    rfm.drop(columns=["segment_id"], errors="ignore").to_csv(
        OUT / "customer_segments.csv", index=False
    )

    return rfm, premium_id


def build_lapse_dataset(customers, orders):
    # Features use strictly pre-cutoff orders.
    before = orders[orders["order_ts"] < CUTOFF].copy()
    completed_before = before[before["status"].eq("completed")].copy()

    cohort_ids = completed_before["customer_id"].dropna().unique()

    # Labels use September completed orders, but never enter the features.
    label_period = orders[
        orders["status"].eq("completed")
        & orders["order_ts"].ge(CUTOFF)
        & orders["order_ts"].lt(LABEL_END)
    ]

    active_in_september = set(label_period["customer_id"].dropna().unique())

    features = completed_before.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
        favourite_category=("category", lambda x: x.mode().iloc[0])
    ).reset_index()

    features["recency"] = (
        CUTOFF - features["last_order_ts"]
    ).dt.days

    final_status_counts = before.groupby("customer_id").agg(
        all_final_orders=("order_id", "nunique"),
        cancelled_orders=(
            "status", lambda x: int(x.eq("cancelled").sum())
        )
    ).reset_index()

    final_status_counts["cancellation_rate"] = (
        final_status_counts["cancelled_orders"]
        / final_status_counts["all_final_orders"]
    )

    features = features.merge(
        final_status_counts[["customer_id", "cancellation_rate"]],
        on="customer_id", how="left"
    )

    customer_features = customers[
        ["customer_id", "city", "age", "gender", "signup_date"]
    ].copy()

    customer_features["tenure_days"] = (
        CUTOFF - customer_features["signup_date"]
    ).dt.days

    features = features.merge(
        customer_features.drop(columns=["signup_date"]),
        on="customer_id", how="left"
    )

    features["lapsed"] = (
        ~features["customer_id"].isin(active_in_september)
    ).astype(int)

    features["favourite_category"] = (
        features["favourite_category"].fillna("unknown").astype(str)
    )
    features["city"] = features["city"].fillna("unknown").astype(str)
    features["gender"] = features["gender"].fillna("unknown").astype(str)

    if features[FEATURES].isna().any().any():
        missing = features[FEATURES].isna().sum()
        raise AssertionError(f"Missing feature values detected:\n{missing}")

    print("\n========== LAPSE COHORT ==========")
    print("Cohort customers:", len(features))
    print("Lapsed customers:", int(features["lapsed"].sum()))
    print("Lapsed percentage:", round(features["lapsed"].mean() * 100, 2))
    print("Feature missing values:", int(features[FEATURES].isna().sum().sum()))

    if len(features) != 1156:
        raise AssertionError(
            f"Expected 1156 lapse cohort customers, found {len(features)}."
        )

    if int(features["lapsed"].sum()) != 443:
        raise AssertionError(
            f"Expected 443 lapsed customers, found {int(features['lapsed'].sum())}."
        )

    return features


def make_pipeline(model):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def report_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    print(f"\n{name}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Lapsed precision:", round(
        precision_score(y_test, predictions, zero_division=0), 4
    ))
    print("Lapsed recall:", round(
        recall_score(y_test, predictions, zero_division=0), 4
    ))
    print("Lapsed F1:", round(
        f1_score(y_test, predictions, zero_division=0), 4
    ))

    return predictions


def train_lapse_models(features, segments):
    data = features.copy()

    X = data[FEATURES]
    y = data["lapsed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_ids = X_train.index
    test_ids = X_test.index

    print("\n========== TRAIN / TEST SPLIT ==========")
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Training class counts:")
    print(y_train.value_counts().sort_index().to_dict())
    print("Test class counts:")
    print(y_test.value_counts().sort_index().to_dict())

    baseline = np.zeros(len(y_test), dtype=int)
    print("\nBaseline: always predict not lapsed")
    print("Accuracy:", round(accuracy_score(y_test, baseline), 4))
    print("Lapsed F1:", round(
        f1_score(y_test, baseline, zero_division=0), 4
    ))

    # Logistic regression grid search.
    lr_pipeline = make_pipeline(LogisticRegression(max_iter=1000))
    lr_grid = GridSearchCV(
        lr_pipeline,
        {"model__C": [0.01, 0.1, 1, 10, 100]},
        cv=5, scoring="f1", n_jobs=-1
    )
    lr_grid.fit(X_train, y_train)

    print("\n========== LOGISTIC REGRESSION ==========")
    print("Best parameters:", lr_grid.best_params_)
    print("Best CV F1:", round(lr_grid.best_score_, 4))
    lr_predictions = report_model(
        "Logistic Regression test results",
        lr_grid.best_estimator_, X_test, y_test
    )

    # Random forest grid search.
    rf_pipeline = make_pipeline(
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    rf_grid = GridSearchCV(
        rf_pipeline,
        {
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [5, 10, 20]
        },
        cv=5, scoring="f1", n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)

    print("\n========== RANDOM FOREST ==========")
    print("Best parameters:", rf_grid.best_params_)
    print("Best CV F1:", round(rf_grid.best_score_, 4))
    rf_predictions = report_model(
        "Random Forest test results",
        rf_grid.best_estimator_, X_test, y_test
    )

    lr_f1 = f1_score(y_test, lr_predictions, zero_division=0)
    rf_f1 = f1_score(y_test, rf_predictions, zero_division=0)

    # Select using cross-validation F1, not test-set performance.
    if rf_grid.best_score_ >= lr_grid.best_score_:
        selected_name = "Random Forest"
        selected_model = rf_grid.best_estimator_
        selected_f1 = rf_f1
        selected_cv_f1 = rf_grid.best_score_
    else:
        selected_name = "Logistic Regression"
        selected_model = lr_grid.best_estimator_
        selected_f1 = lr_f1
        selected_cv_f1 = lr_grid.best_score_

    print("\nSelected model:", selected_name)
    print("Selected CV F1:", round(selected_cv_f1, 4))
    print("Selected test F1:", round(selected_f1, 4))

    if selected_f1 < 0.55:
        raise AssertionError(
            "Selected model test F1 is below the assignment minimum of 0.55."
        )

    # Show scaler mean fitted only on training data.
    fitted_preprocessor = lr_grid.best_estimator_.named_steps["preprocessor"]
    fitted_scaler = (
        fitted_preprocessor.named_transformers_["numeric"]
        .named_steps["scaler"]
    )
    frequency_index = NUMERIC.index("frequency")

    print("\nTraining-only scaler frequency mean:",
          round(fitted_scaler.mean_[frequency_index], 4))
    print("Full-cohort frequency mean:",
          round(data["frequency"].mean(), 4))
    print(
        "These differ because the scaler was fitted only on training rows, "
        "not on the full cohort."
    )

    # Feature importance from the tuned random forest.
    rf_preprocessor = rf_grid.best_estimator_.named_steps["preprocessor"]
    rf_model = rf_grid.best_estimator_.named_steps["model"]
    rf_names = rf_preprocessor.get_feature_names_out()
    importance = pd.DataFrame({
        "feature": rf_names,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nTop five Random Forest features:")
    print(importance.head(5).to_string(index=False))

    # Logistic regression coefficient interpretation.
    lr_preprocessor = lr_grid.best_estimator_.named_steps["preprocessor"]
    lr_model = lr_grid.best_estimator_.named_steps["model"]
    lr_names = lr_preprocessor.get_feature_names_out()
    coefficients = pd.DataFrame({
        "feature": lr_names,
        "coefficient": lr_model.coef_[0]
    })
    coefficients["absolute"] = coefficients["coefficient"].abs()
    coefficients = coefficients.sort_values("absolute", ascending=False)

    print("\nThree Logistic Regression coefficients:")
    print(coefficients[["feature", "coefficient"]].head(3).to_string(index=False))

    # Oversample minority lapsed rows in training data only.
    train_data = X_train.copy()
    train_data["_target"] = y_train

    majority_count = int(y_train.value_counts().max())
    minority_count = int(y_train.value_counts().min())
    minority_class = int(y_train.value_counts().idxmin())
    needed = majority_count - minority_count

    minority_rows = train_data[train_data["_target"].eq(minority_class)]
    duplicated = minority_rows.sample(
        n=needed, replace=True, random_state=42
    )

    balanced = pd.concat([train_data, duplicated], ignore_index=True)
    X_balanced = balanced[FEATURES]
    y_balanced = balanced["_target"]

    balanced_lr = clone(lr_grid.best_estimator_)
    balanced_lr.fit(X_balanced, y_balanced)

    balanced_predictions = balanced_lr.predict(X_test)

    print("\n========== TRAINING-ONLY OVERSAMPLING ==========")
    print("Original training counts:", y_train.value_counts().to_dict())
    print("Balanced training counts:", y_balanced.value_counts().to_dict())
    print("Original LR precision:",
          round(precision_score(y_test, lr_predictions, zero_division=0), 4))
    print("Balanced LR precision:",
          round(precision_score(y_test, balanced_predictions, zero_division=0), 4))
    print("Original LR recall:",
          round(recall_score(y_test, lr_predictions, zero_division=0), 4))
    print("Balanced LR recall:",
          round(recall_score(y_test, balanced_predictions, zero_division=0), 4))
    print("Original LR F1:", round(lr_f1, 4))
    print("Balanced LR F1:",
          round(f1_score(y_test, balanced_predictions, zero_division=0), 4))

    # Persist selected model and verify exact prediction reproduction.
    model_path = OUT / "lapse_model.joblib"
    joblib.dump(selected_model, model_path)
    reloaded = joblib.load(model_path)

    original_predictions = selected_model.predict(X_test)
    reloaded_predictions = reloaded.predict(X_test)

    same = np.array_equal(original_predictions, reloaded_predictions)
    print("\nReloaded model reproduces test predictions:", same)

    if not same:
        raise AssertionError("Reloaded model predictions do not match.")

    # Rank Premium test customers using the selected model's probabilities.
    if not hasattr(selected_model, "predict_proba"):
        raise AssertionError("Selected model does not support predict_proba.")

    probabilities = selected_model.predict_proba(X_test)[:, 1]

    test_prediction_frame = pd.DataFrame({
        "customer_id": data.loc[X_test.index, "customer_id"].values,
        "lapse_probability": probabilities,
        "actually_lapsed": y_test.values
    })

    premium_ids = set(
        segments.loc[segments["is_premium"], "customer_id"]
    )

    retention = test_prediction_frame[
        test_prediction_frame["customer_id"].isin(premium_ids)
    ].sort_values("lapse_probability", ascending=False).head(10)

    retention = retention[
        ["customer_id", "lapse_probability", "actually_lapsed"]
    ]

    retention.to_csv(OUT / "retention_list.csv", index=False)

    print("\n========== PREMIUM RETENTION LIST ==========")
    print(retention.to_string(index=False))
    print("Customers in list who truly lapsed:",
          int(retention["actually_lapsed"].sum()))

    metrics = {
        "cohort_customers": int(len(data)),
        "lapsed_customers": int(data["lapsed"].sum()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "logistic_best_params": lr_grid.best_params_,
        "logistic_cv_f1": float(lr_grid.best_score_),
        "logistic_test_f1": float(lr_f1),
        "random_forest_best_params": rf_grid.best_params_,
        "random_forest_cv_f1": float(rf_grid.best_score_),
        "random_forest_test_f1": float(rf_f1),
        "selected_model": selected_name,
        "selected_test_f1": float(selected_f1),
        "reloaded_predictions_match": bool(same),
        "premium_retention_list_true_lapsed": int(
            retention["actually_lapsed"].sum()
        )
    }

    with open(OUT / "part_b_customer_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nPart B customer analysis completed.")
    print("Created:")
    print(" - outputs/customer_segments.csv")
    print(" - outputs/lapse_model.joblib")
    print(" - outputs/retention_list.csv")
    print(" - outputs/part_b_kmeans_evaluation.csv")
    print(" - outputs/part_b_segment_profile.csv")
    print(" - outputs/part_b_customer_metrics.json")


def main():
    customers, orders = load_data()
    rfm = build_rfm(customers, orders)
    segments, premium_id = cluster_customers(rfm)
    lapse_data = build_lapse_dataset(customers, orders)
    train_lapse_models(lapse_data, segments)


if __name__ == "__main__":
    main()
