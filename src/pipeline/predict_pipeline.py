import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifact\model.pkl'
            preprocessor_path = 'artifact\preprocessor.pkl'
            preprocessor= load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)
            data_scaled = preprocessor.transform(features)
            model.predict(data_scaled)
            return preds
        except Exception as e:
            raise(e,sys)

class CustomData:
    def __init__(self, time_spent_on_course, number_of_video_watched, number_of__taken, quiz_scores, completion_rate, device_type, course_category):
        self.time_spent_on_course= time_spent_on_course
        self.number_of_video_watched = number_of_video_watched
        self.number_of__taken = number_of__taken
        self.quiz_scores = quiz_scores
        self.completion_rate = completion_rate
        self.device_type = device_type
        self.course_category = course_category

    def get_data_as_data_frame(self):
        'TimeSpentOnCourse '=  self.time_spent_on_course
        'NumberOfVideoWatched '= self.number_of_video_watched
        'NumberOfTaken' = self.number_of__taken
        'QuizScores' = self.quiz_scores
        'CompletionRate' = self.completion_rate
        'DeviceType '= self.device_type
       ' CourseCategory' = self.course_category







