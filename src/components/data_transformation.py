import sys
import numpy as np
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_obj(self):
        """
        This function transform the dataset (Categorical data, Scaling)
        :return: the preprocessor as ColumnTransformer obj
        """
        try:
            numerical_columns =[
                'TimeSpentOnCourse',
                'NumberOfVideosWatched',
                'NumberOfQuizzesTaken',
                'QuizScores',
                'CompletionRate',
                'DeviceType'
            ]
            categorical_columns = ['CourseCategory']

            num_pipeline = Pipeline(
                steps=[
                    ('scaler', MaxAbsScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', MaxAbsScaler())

                ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
               [
                   ('numerical_pipeline', num_pipeline, numerical_columns),
                   ('categorical_pipeline', cat_pipeline, categorical_columns)
               ]

            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("The train and test data was read")
            logging.info("get preprocessing object")

            preprocessor_obj = self.get_transformer_obj()

            drop_column = 'UserID'
            target_column_name = 'CourseCompletion'
            numerical_columns = [
                'TimeSpentOnCourse',
                'NumberOfVideosWatched',
                'NumberOfQuizzesTaken',
                'QuizScores',
                'CompletionRate',
                'DeviceType',

            ]
            categorical_columns = ['CourseCategory']
            input_feature_train_df = train_df.drop(columns=[target_column_name, drop_column], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name, drop_column], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)


