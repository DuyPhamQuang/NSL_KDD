import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
import seaborn as sns
from time import time

class DataPreprocessor:
    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        Initializes the DataPreprocessor with training and test datasets.

        Parameters
        ----------
        train_data : pd.DataFrame
            The training dataset.
        test_data : pd.DataFrame
            The test dataset.
        """
        self.train_data = train_data
        self.test_data = test_data

    def preprocess_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Loads the NSL-KDD dataset, preprocesses it, and returns the preprocessed datasets.

        The function first loads the dataset from the 'NSL_KDD_Dataset' folder, and then defines the column names. It then
        preprocesses the dataset by one-hot-encoding categorical features and normalizing numeric features. Finally, it returns
        the preprocessed datasets.

        Returns
        -------
        tuple of pd.DataFrame
            The preprocessed training and test datasets.
        """

        # Define column names
        columns = (['duration'
        ,'protocol_type'
        ,'service'
        ,'flag'
        ,'src_bytes'
        ,'dst_bytes'
        ,'land'
        ,'wrong_fragment'
        ,'urgent'
        ,'hot'
        ,'num_failed_logins'
        ,'logged_in'
        ,'num_compromised'
        ,'root_shell'
        ,'su_attempted'
        ,'num_root'
        ,'num_file_creations'
        ,'num_shells'
        ,'num_access_files'
        ,'num_outbound_cmds'
        ,'is_host_login'
        ,'is_guest_login'
        ,'count'
        ,'srv_count'
        ,'serror_rate'
        ,'srv_serror_rate'
        ,'rerror_rate'
        ,'srv_rerror_rate'
        ,'same_srv_rate'
        ,'diff_srv_rate'
        ,'srv_diff_host_rate'
        ,'dst_host_count'
        ,'dst_host_srv_count'
        ,'dst_host_same_srv_rate'
        ,'dst_host_diff_srv_rate'
        ,'dst_host_same_src_port_rate'
        ,'dst_host_srv_diff_host_rate'
        ,'dst_host_serror_rate'
        ,'dst_host_srv_serror_rate'
        ,'dst_host_rerror_rate'
        ,'dst_host_srv_rerror_rate'
        ,'class'
        ,'difficulty_level'])

        self.train_data.columns = columns
        self.test_data.columns = columns

        ''' Handle categorical features '''

        # Filter out columns with type object
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns
        categorical_cols_train = [col for col in categorical_cols if col != 'class']  #Exclude class column

        categorical_cols_test = [col for col in categorical_cols if col in self.test_data.columns]


        ''' One-hot-Encoding '''

        # Initialize and fit OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(self.train_data[categorical_cols_train])

        # Transform categorical columns
        encoded_train_data = encoder.transform(self.train_data[categorical_cols_train])
        encoded_feature_names_train = encoder.get_feature_names_out(categorical_cols_train)
        encoded_df_train = pd.DataFrame(encoded_train_data, columns=encoded_feature_names_train, index=self.train_data.index)

        encoded_test_data = encoder.transform(self.test_data[categorical_cols_test])
        encoded_feature_names_test = encoder.get_feature_names_out(categorical_cols_test)
        encoded_df_test = pd.DataFrame(encoded_test_data, columns=encoded_feature_names_test, index=self.test_data.index)

        # Combine with numeric features excluding the difficulty level
        numeric_cols_train = [col for col in self.train_data.columns if col not in categorical_cols_train and col != 'difficulty_level']
        numeric_cols_test = [col for col in self.test_data.columns if col not in categorical_cols_test and col != 'class' 
                        and col != 'difficulty_level']
        
        train_data_encoded = pd.concat([self.train_data[numeric_cols_train], encoded_df_train], axis=1)
        test_data_encoded = pd.concat([self.test_data[numeric_cols_test], encoded_df_test], axis=1)

        number_of_features_train = train_data_encoded.shape[1]

        # Add missing columns to test data and reorder to match train data
        train_columns = train_data_encoded.columns
        for col in train_columns:
            if col not in test_data_encoded.columns:
                test_data_encoded[col] = 0
        test_data_encoded = test_data_encoded[train_columns]

        '''Normalize Numeric Features'''

        scaler = MinMaxScaler()
        scaler.fit(train_data_encoded[numeric_cols_train])

        numeric_cols_train = train_data_encoded.select_dtypes(include=['int64', 'float64']).columns
        train_data_encoded[numeric_cols_train] = scaler.transform(train_data_encoded[numeric_cols_train])

        numeric_cols_test = test_data_encoded.select_dtypes(include=['int64', 'float64']).columns
        test_data_encoded[numeric_cols_test] = scaler.transform(test_data_encoded[numeric_cols_test])

        return train_data_encoded, test_data_encoded, number_of_features_train
    
def main():
    # Load datasets
    train_data = pd.read_csv('NSL_KDD_Dataset/KDDTrain+.txt', header=None)
    test_data = pd.read_csv('NSL_KDD_Dataset/KDDTest+.txt', header=None)

    # Initialize preprocessor
    preprocessor = DataPreprocessor(train_data, test_data)

    # Preprocess datasets
    train_data_encoded, test_data_encoded = preprocessor.preprocess_datasets()

    print("Preprocessing complete.")
    print(f"Train data shape: {train_data_encoded.shape}")
    print(f"Test data shape: {test_data_encoded.shape}")

if __name__ == "__main__":
    main()

