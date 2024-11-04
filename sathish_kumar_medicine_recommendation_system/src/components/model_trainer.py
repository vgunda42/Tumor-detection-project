import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from exception import CustomException
from logger import logging

from utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                "SVM": SVC(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBClassifier(),
                "KNeighbors": KNeighborsClassifier(),
                "MultinomialNB": MultinomialNB()
            }

            # params = {
            #     "Decision Tree": {
            #         'criterion': ['gini', 'entropy'],
            #         'splitter': ['best', 'random'],
            #         'max_depth': [None, 10, 20, 30],
            #     },
            #     "Random Forest": {
            #         'n_estimators': [50, 100, 150],
            #         'max_depth': [None, 10, 20, 30],
            #     },
            #     "Gradient Boosting": {
            #         'learning_rate': [0.1, 0.01, 0.05],
            #         'n_estimators': [50, 100, 150],
            #     },
            #     "Logistic Regression": {
            #         'C': [0.1, 1, 10],
            #         'penalty': ['l2'],
            #         'solver': ['lbfgs', 'liblinear'],
            #     },
            #     "XGBClassifier": {
            #         'learning_rate': [0.1, 0.01, 0.05],
            #         'n_estimators': [50, 100, 150],
            #         'max_depth': [3, 5, 7],
            #     },
            #     "SVM": {
            #         'C': [0.1, 1, 10],
            #         'kernel': ['linear', 'rbf'],
            #         'gamma': ['scale', 'auto']
            #     },
            #     "KNeighbors": {
            #         'n_neighbors': [3, 5, 7, 9],
            #         'weights': ['uniform', 'distance']
            #     },
            #     "MultinomialNB": {
            #         'alpha': [0.1, 0.5, 1.0]  
            #     }
            # }

            # model_report: dict = evaluate_models(
            #     X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            #     models=models, param=params
            # )

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models
            )

            # Get the best model based on highest accuracy
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found")

            logging.info(f"Best model found with accuracy: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Evaluate the best model on the test set
            predictions = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
