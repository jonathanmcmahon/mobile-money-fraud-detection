"""A brief exploration of a synthetic financial dataset for fraud detection.

Kaggle dataset: https://www.kaggle.com/ntnu-testimon/paysim1/data
"""
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def configure_logging(logfile):
    """Sets up logging facilities and returns a logger."""

    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)

    # Create file handler
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def analyze_dataset(df, logger):
    """Prints summary info for a dataset."""

    total_transactions = df.count()[0]
    total_fraud = df["isFraud"].value_counts()
    logger.info(f"Total transactions: {total_transactions}, total fraud: {total_fraud[1]}")

    unique_src = df["nameOrig"].nunique()
    unique_dst = df["nameDest"].nunique()
    logger.info(f"Unique src customer: {unique_src}, unique dst customer: {unique_dst}")


def build_features(df, logger):
    """Build dataset features prior to model training."""

    # Drop source and dest customer name (categorical with extremely high number of categories)
    df = df.drop(labels=["nameOrig","nameDest", "isFlaggedFraud"], axis=1)

    # One-hot encode the type
    onehot_type = pd.get_dummies(df["type"], prefix="type")
    df = df.join(onehot_type)
    df = df.drop(labels="type", axis=1)

    # Add hour of day - note that this assumes step 0 in original data corresponds with 00:00
    hour = df["step"] % 24
    hour = hour.rename("hour")
    df = df.join(hour)

    return df


def split_ts_data(df, n_splits, y_label, logger):
    """Split a timeseries dataset into multiple datasets for cross-validation."""

    # Split dataset into x and y -- x contains the features, y is the label
    y = df[y_label]
    x = df.drop(labels=y_label, axis=1)

    # Get indices for a timeseries split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    # print(tscv)

    # Build datasets from split indices
    datasets = []
    for train_index, test_index in tscv.split(x):
        d = {}
        # print("TRAIN:", train_index, "TEST:", test_index)
        d["x_train"] = x.iloc[train_index]
        d["y_train"] = y.iloc[train_index]
        d["x_test"] = x.iloc[test_index]
        d["y_test"] = y.iloc[test_index]
        datasets.append(d)

    # Summarize splits
    logger.info("### Summary of split datasets ###")
    for idx, s in enumerate(datasets):
        logger.info(f"Dataset {idx}")
        x_train_rows, y_train_rows = s["x_train"].shape[0], s["y_train"].shape[0]
        x_test_rows, y_test_rows = s["x_test"].shape[0], s["y_test"].shape[0]
        assert s["x_train"].shape[0] == s["y_train"].shape[0]
        assert s["x_test"].shape[0] == s["y_test"].shape[0]
        logger.info(f"    Training records: {x_train_rows}")
        logger.info(f"    Test records:     {x_test_rows}")
        logger.info(f"    TOTAL: {x_train_rows + x_test_rows}")

    return datasets


def train_model(data, ModelClass):
    """Train and return a model."""
    model = ModelClass()
    model.fit(data["x_train"], data["y_train"])
    return model


def eval_model(model, data, logger, plots=False):
    """Evaluate a model against test data."""

    # Rename logger
    model_name = model.__class__.__name__
    saved_logger_name = logger.name
    logger.name = model_name

    # Evaluate model against test data
    y_pred = model.predict(data["x_test"])

    # Plot predictions
    if plots:
        plt.scatter(np.arange(y_pred.shape[0]), y_pred, marker=".")
        plt.show()

    # Calculate accuracy
    accuracy = accuracy_score(data["y_test"], y_pred)
    logger.info("Accuracy: %.2f%%" % (accuracy * 100.0))

     # Calculate F1 score
    f1score = f1_score(data["y_test"], y_pred)
    logger.info("F1 Score: %.2f%%" % f1score)

    # Calculate confusion matrix
    cm = confusion_matrix(data["y_test"], y_pred)
    logger.info("Confusion matrix:")
    logger.info(f"TN: {cm[0][0]}\t FP: {cm[0][1]}")
    logger.info(f"FN: {cm[1][0]}\t TP: {cm[1][1]}")

    # Plot confusion matrix
    if plots:
        plt.pie(cm.flatten())
        plt.show()

    # Rename logger to original name
    logger.name = saved_logger_name

    return (accuracy, f1score, cm)


if __name__ == "__main__":

    logfile = "model.log"

    # Set up logging facilities
    logger = configure_logging(logfile=logfile)

    # Read dataset into pandas dataframe
    df = pd.read_csv("data/data-train.csv")

    # Print summary analysis of dataset
    analyze_dataset(df, logger=logger)

    # Build features
    df = build_features(df, logger=logger)

    # Split into multiple train/validation sets
    all_sets = split_ts_data(df, n_splits=4, y_label="isFraud", logger=logger)

    # Iterate over train/validation sets
    for data in all_sets:

        # Train model on train data
        model = train_model(data=data, ModelClass=RandomForestClassifier)

        # Evaluate model against test data
        acc, f1, cm = eval_model(model, data, logger)

        # Plot feature importance
        features = data["x_train"].columns
        importance = model.feature_importances_
        indices = np.argsort(importance)
        plt.figure(1)
        plt.title("Feature Importances")
        plt.barh(range(len(indices)), importance[indices], color="b", align="center")
        plt.yticks(range(len(indices)), features[indices])
        plt.xlabel("Relative Importance")
        plt.show()

    print("Model training and evaluation is complete.")
    print(f"Results are logged to {logfile}")
