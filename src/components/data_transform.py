import os, sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This is responsible for data transformation
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            numerical_pipeline = Pipeline(
                steps=[
                   ("imputer", SimpleImputer(strategy="median")),
                   ("scaler", StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean = False))
                ]
            )
            logging.info("Numerical columns standardization completed")
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)  
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            preprocessor = self.get_data_transformer_object()
            target_col = "math_score"
            train_df_x = train_df.drop(columns=[target_col], axis=1)
            train_df_y = train_df[target_col]
            test_df_x = test_df.drop(columns=[target_col], axis=1)
            test_df_y = test_df[target_col]

            logging.info("Applying preprocessor object on train_df and test_df")

            train_x = preprocessor.fit_transform(train_df_x)
            test_x = preprocessor.transform(test_df_x)

            train_arr = np.c_[train_x, train_df_y]
            test_arr = np.c_[test_x, test_df_y]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("Saved preprocessor object")

            return (
                train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)