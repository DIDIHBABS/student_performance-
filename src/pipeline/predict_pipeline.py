
import os
import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path=os.path.join("artifact","model.pkl")
            preprocessor_path=os.path.join('artifact','preprocessor.pkl')
            print("Before Loading")
            preprocessor= load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 time_spent_on_course,
                 number_of_video_watched,
                 number_of_quiz,
                 quiz_scores,
                 completion_rate,
                 device_type,
                 course_category):
        self.time_spent_on_course= time_spent_on_course
        self.number_of_video_watched = number_of_video_watched
        self.number_of_quiz = number_of_quiz
        self.quiz_scores = quiz_scores
        self.completion_rate = completion_rate
        self.device_type = device_type
        self.course_category = course_category


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'TimeSpentOnCourse': [self.time_spent_on_course],
                'NumberOfVideoWatched ': [self.number_of_video_watched],
                'NumberOfQuizzesTaken': [self.number_of_quiz],
                'QuizScores': [self.quiz_scores],
                'CompletionRate': [self.completion_rate],
                'DeviceType ': [self.device_type],
                'CourseCategory': [self.course_category],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)







