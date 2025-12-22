import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main(data_dir, n_estimators, max_depth):
    X_train = joblib.load(os.path.join(data_dir, "X_train.pkl"))
    X_test  = joblib.load(os.path.join(data_dir, "X_test.pkl"))
    y_train = joblib.load(os.path.join(data_dir, "y_train.pkl"))
    y_test  = joblib.load(os.path.join(data_dir, "y_test.pkl"))

    mlflow.set_experiment("RF-CI-Project")

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, "model")

        cm = confusion_matrix(y_test, y_pred)
        plt.imshow(cm)
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    main(args.data_dir, args.n_estimators, args.max_depth)
