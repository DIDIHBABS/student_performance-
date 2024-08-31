import os
import sys
from dataclasses import dataclass

 # Algorithms
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


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
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(),
            }

            # Define hyperparameters for grid search
            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    # 'splitter': ['best', 'random'],
                    # 'max_features': ['sqrt', 'log2'],
                },
                "Random Forest": {
                    # 'criterion': ['gini', 'entropy', 'log_loss'],
                    # 'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    # 'loss': ['log_loss', 'deviance', 'exponential'],
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion': ['friedman_mse', 'squared_error'],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Logistic Regression": {
                    'C': [0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga']
                },
                "XGBClassifier": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            #                                      models=models )
            report, accuracy_score_train, confusion_matrix_train = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                                                   models=models, params=params)
            print(accuracy_score_train)
            ## To get best model score from dict
            best_model_score = max(sorted(accuracy_score_train.values()))

            ## To get best model name from dict

            best_model_name = list(accuracy_score_train.keys())[
                list(accuracy_score_train.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.8:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset are{best_model_name }")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            class_report = classification_report(y_test, predicted)
            test_accuracy_score = accuracy_score(y_test, predicted)
            test_confusion_matrix = confusion_matrix(y_test, predicted)

            return class_report, test_accuracy_score, test_confusion_matrix

        except Exception as e:
            raise CustomException(e, sys)