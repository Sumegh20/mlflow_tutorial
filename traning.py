import argparse
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha')
    parser.add_argument('--l1_ratio', type=float, default=0.5, help='l1_ratio')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Read the wine-quality csv file from the URL
    csv_url = "winequality-red.csv"
    try:
        data = pd.read_csv(csv_url)
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality"
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    parameters = parse_opt()

    alpha = parameters.alpha
    l1_ratio = parameters.l1_ratio

    # mlflow.set_tracking_uri("https://dagshub.com/Sumegh20/mlflow_test_1.mlflow")
    mlflow.set_experiment("red_wine_quality")

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # model parameters
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        # model metric
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)

        mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticNet", signature=signature)
        # mlflow.sklearn.log_model(lr, "model", signature=signature)
