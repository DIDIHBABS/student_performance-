import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

    def evaluate_models(X_train, y_train, X_test, y_test, models, param):
        try:
            report = {}
            accuracy_score_train = {}
            confusion_matrix_train = {}
            for i in range(len(list(models))):
                model = list(models.values())[i]
                model.fit(X_train, y_train)

                # gs = GridSearchCV(model, para, cv=3)
                # gs.fit(X_train, y_train)
                #
                # model.set_params(**gs.best_params_)


                # model.fit(X_train, y_train)  # Train model

                # Make predictions

                y_train_pred = model.predict(X_train)

                y_test_pred = model.predict(X_test)

                # Evaluate Train and Test dataset
                train_model_score = classification_report(y_train, y_train_pred)
                test_model_score = classification_report(y_test, y_test_pred)

                train_confusion_matrix= confusion_matrix(y_train, y_train_pred)
                test_confusion_matrix = confusion_matrix(y_test, y_test_pred)

                train_accuracy_score = accuracy_score(y_train, y_train_pred)
                test_accuracy_score = accuracy_score(y_test, y_test_pred)

                report[list(models.keys())[i]] = test_model_score
                accuracy_score_train[list(models.keys())[i]] = test_accuracy_score
                confusion_matrix_train[list(models.keys())[i]] = test_confusion_matrix
            return report, accuracy_score_train, confusion_matrix_train

        except Exception as e:
            raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)