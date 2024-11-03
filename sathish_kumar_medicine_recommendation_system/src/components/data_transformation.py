import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "label_encoded_data.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.label_encoder = LabelEncoder()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded train and test data.")

            target_column_name = "prognosis"

            # Separate input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Encode target labels
            logging.info("Encoding target labels with LabelEncoder.")
            target_feature_train_df = self.label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_df = self.label_encoder.transform(target_feature_test_df)

            # Combine input features and encoded target for train and test arrays
            train_arr = np.c_[input_feature_train_df.values, target_feature_train_df]
            test_arr = np.c_[input_feature_test_df.values, target_feature_test_df]

            # Save the LabelEncoder object
            logging.info("Saving LabelEncoder object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=self.label_encoder
            )

            # Retrieve class mapping
            class_mapping = self.label_encoder.classes_
            logging.info(f"Classes mapping: {class_mapping}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                class_mapping  
            )
        
        except Exception as e:
            raise CustomException(e, sys)


