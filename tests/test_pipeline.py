import pandas as pd

from pipeline import (
    FEATURE_COLS,
    TARGET_COL,
    TRAFFIC_LABELS,
    TRAFFIC_MAP,
    evaluate,
    load_data,
    train_model,
)


def test_dataset_has_required_columns():
    df = load_data()
    missing = set(FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_features_are_numeric_after_encoding():
    df = load_data()
    for col in FEATURE_COLS:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"


def test_traffic_codes_match_form_labels():
    # The dispatch form and the training data must use the same coding.
    assert set(TRAFFIC_MAP.values()) == set(TRAFFIC_LABELS.keys())


def test_model_trains_and_predicts_known_labels():
    df = load_data()
    clf, X_test, y_test = train_model(df, n_estimators=20)
    preds = clf.predict(X_test.iloc[:10])
    assert set(preds) <= set(df[TARGET_COL].unique())


def test_evaluation_returns_valid_scores():
    df = load_data()
    clf, X_test, y_test = train_model(df, n_estimators=20)
    scores = evaluate(clf, X_test, y_test)
    assert 0.0 <= scores["accuracy"] <= 1.0
    assert 0.0 <= scores["macro_f1"] <= 1.0
