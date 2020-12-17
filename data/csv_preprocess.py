import glob
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def normalize_defected(df: pd.DataFrame, img_height=512, img_width=512, angle_max=np.pi) -> pd.DataFrame:
    """
    Numerical normalization before creating pipeline
    :param df: Defected pd.Dataframe object to normalize
    :param img_height: Image height of all dataset
    :param img_width: Image width of all dataset
    :param angle_max: Max angle in radian
    :return: Normalized dataframe
    """
    # Normalizing semi axes with half of image shape
    df['semi_major'] /= (img_height / 2)
    df['semi_minor'] /= (img_width / 2)
    # Normalizing center points wit image shape
    df['x_to_center'] /= img_width
    df['y_to_center'] /= img_height
    # Converting -pi<x<=+pi scale into 0<x<=+2pi, then normalizing between 0-1
    df['rot_angle'] = (df['rot_angle'] + angle_max) / (2 * angle_max)
    return df


def one_hot_encode(df: pd.DataFrame, column='class') -> pd.DataFrame:
    """
    Encoding multi-class labeled column into one-hot encoded columns
    :param df: Combined pd.Dataframe object
    :param column: Selected column name
    :return: One-hot encoded dataframe
    """
    # One-hot encoder object created with auto category configuration
    # drop='first' can be enable to get rid of dummy variable
    # ohe = OneHotEncoder(categories='auto', drop='first')
    ohe = OneHotEncoder(categories='auto')
    # Reshaping data to be able to fit 'ohe'
    values = df[column].values.reshape(-1, 1)
    # Encoding labels
    encoded = ohe.fit_transform(values).toarray().astype(np.uint8)
    # Reading column index
    id_df = df.columns.get_loc(column)
    # Inserting encoded labels into dataframe with same name
    for binary_level in range(encoded.shape[1]):
        df.insert(id_df + binary_level, f'{column}_{binary_level}', encoded[:, binary_level])
    # Removing useless column
    df = df.drop(columns=[column])
    return df


class DatasetDataframe:
    def __init__(self, data_root_path='data/raw', shuffle=False, random_seed=10):
        """
        Task related dataset class in order to manipulate dataset information and preprocess data
        before feeding neural network.
        :param data_root_path: Path to raw data folder
        :param random_seed: Random seed for shuffling dataset
        :param shuffle: Shuffles dataset if set True
        """
        # Attributes
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.data_root_path = data_root_path
        # Column names
        self.col_dict = {
            'old_cols': ['semi_major', 'semi_minor', 'rot_angle', 'x_to_center', 'y_to_center'],
            'new_cols': ['path', 'class', 'is_defected', 'is_test'],
        }
        self.total_cols = self.col_dict['new_cols'] + self.col_dict['old_cols']

        self.__normal_df = self.create_normal_df()
        self.__defected_df = self.create_defected_df()

    def create_defected_df(self) -> pd.DataFrame:
        """
        Creates defected *.csv dataset to processing easier on later steps.
        :return: pd.Dataframe object of concatenated datasets
        """
        # Listing all *.txt files to get older information
        txt_files = sorted(glob.glob('**/*.txt'))
        # Placeholder for defected dataset
        temp_df = []
        for class_id, file in enumerate(txt_files):
            # Path to specific class folder
            class_root_path = os.path.join(self.data_root_path, file.split('.')[0].split('/')[1])
            # Reading *.txt file as csv and removing false rows at the end
            df = pd.read_csv(file, sep='\t', header=None, index_col=0).iloc[:-1]
            # Sorting semi axis lengths, semi major axis has to be longer then minor one
            semi_vals = df.iloc[:, :2].values
            df.iloc[:, :2] = np.array([(y, x) if x < y else (x, y) for x, y in semi_vals])
            # Renaming column names
            df.columns = self.col_dict['old_cols']
            # Creating class define column
            class_col = pd.Series([class_id + 1] * len(df), index=df.index)
            # Creating file path column
            paths = sorted(glob.glob(os.path.join(class_root_path, '*.png')))
            path_col = pd.Series(paths, index=df.index)
            # Creating test column to choose which data will be tested after training
            # First 120 sample will be used for training, the rest for testing
            is_test = np.arange(0, 150)
            is_test[is_test < 120] = 0
            is_test[is_test != 0] = 1
            is_test_col = pd.Series(is_test, index=df.index)
            # Creating defect column to classify easier while training
            if 'def' in file:
                is_defected_col = pd.Series([1] * len(df), index=df.index)
            else:
                is_defected_col = pd.Series([0] * len(df), index=df.index)

            # Appending all new columns into existing one to produce a new Dataframe
            new_df_list = [path_col, class_col, is_defected_col, is_test_col]
            for col_name, col_data in zip(self.col_dict['new_cols'], new_df_list):
                df[col_name] = col_data
            # Changing order of columns
            df = df[self.total_cols]
            # Every class will be appended into placeholder list
            temp_df.append(df)

        # All dataframes will be concatenated on sample axis and
        # combined Dataframe will be returned
        df = pd.DataFrame(pd.concat(temp_df, axis=0))
        # Removing old index column
        df = df.reset_index().iloc[:, 1:]
        return df

    def create_normal_df(self) -> pd.DataFrame:
        """
        Creates normal *.csv dataset to processing easier on later steps.
        :return: pd.Dataframe object of concatenated datasets
        """
        normal_dirs = sorted([dir_path for dir_path in os.listdir(self.data_root_path) if 'norm' in dir_path])
        # Placeholder for normal dataset
        temp_df = []
        for class_id, class_dir in enumerate(normal_dirs):
            # Path to specific class folder
            class_root_folder = os.path.join(self.data_root_path, class_dir)
            # Creating file path column
            paths = sorted(glob.glob(os.path.join(class_root_folder, '*.png')))
            # Limiting normal dataset with 200 images to not to bias model so much
            paths = paths[:150]
            path_col = pd.Series(paths)
            # Creating class define column
            class_col = pd.Series([class_id + 1] * len(paths))
            # Creating test column to choose which data will be tested after training
            # First 150 sample will be used for training, the rest for testing
            is_test = np.arange(0, 150)
            is_test[is_test < 120] = 0
            is_test[is_test != 0] = 1
            is_test_col = pd.Series(is_test)
            # Creating defect column to classify easier while training
            if 'norm' in class_dir:
                is_defected_col = pd.Series([0] * len(paths))
            else:
                is_defected_col = pd.Series([1] * len(paths))
            df = pd.DataFrame(pd.concat([path_col, class_col, is_defected_col, is_test_col], axis=1))
            df[self.col_dict['old_cols']] = 0
            # Changing order of columns
            df.columns = self.total_cols
            # Every class will be appended into placeholder list
            temp_df.append(df)
        # All dataframes will be concatenated on sample axis and
        # combined Dataframe will be returned
        df = pd.DataFrame(pd.concat(temp_df, axis=0))
        # Removing old index column
        df = df.reset_index().iloc[:, 1:]
        return df

    @property
    def normal_df(self):
        """Returns normal dataset"""
        return self.__normal_df

    @property
    def defected_df(self):
        """Returns defected dataset"""
        return self.__defected_df


if __name__ == '__main__':
    # Creating an instance from DatasetDataframe object
    dataset = DatasetDataframe(data_root_path='data/raw', random_seed=10, shuffle=True)
    # Reading dataframe properties from object
    normal_df = dataset.normal_df
    defected_df = dataset.defected_df
    # Normalizing numerical values in defected dataframe
    defected_df = normalize_defected(defected_df)
    # Combining both normal and defected dataframes on sample axis
    combined_df = pd.concat([normal_df, defected_df])
    # One-hot encoding on 'class' column, dummy variable will be removed inside of the function
    combined_df = one_hot_encode(combined_df, column='class')
    # Splitting into train and test
    train_df = combined_df[combined_df['is_test'] == 0]
    test_df = combined_df[combined_df['is_test'] == 1]
    # Removing 'is_test' column, it's not useful anymore
    train_df = train_df.drop(columns=['is_test'])
    test_df = test_df.drop(columns=['is_test'])
    # Dataset is shuffled if specified
    if dataset.shuffle:
        # Seed given into shuffle to reproduce the same state
        train_df = train_df.sample(frac=1, random_state=dataset.random_seed).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=dataset.random_seed).reset_index(drop=True)
    # Creates directory to save results
    if not os.path.exists('data/csv'):
        os.mkdir('data/csv')
    # Saving into *.csv
    train_df.to_csv('data/csv/train.csv', index=False, sep=',', encoding='utf-8', header=False)
    test_df.to_csv('data/csv/test.csv', index=False, sep=',', encoding='utf-8', header=False)
