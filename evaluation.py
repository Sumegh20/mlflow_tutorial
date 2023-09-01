import mlflow, warnings
import pandas as pd

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Read the wine-quality csv file from the URL
    csv_url = "winequality-red.csv"
    
    data = pd.read_csv(csv_url)

    model_name = "ElasticNet"
    stage = "Production"

    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{stage}"
    )

    y_pred = model.predict(data.iloc[0:1])

    print(y_pred)