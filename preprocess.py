import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


class DataPreprocess:
    def __init__(self, seed=42):
        self.df = None
        self.seed = seed
    
    def __drop__(self, data):
        return data.drop(["salary", "salary_currency", "employee_residence", "company_location", "job_title"], axis=1)
    
    def __descretize_salary__(self, row):
        if row["salary_in_usd"] > 200000:
            row["salary_in_usd"] = 1
        else:
            row["salary_in_usd"] = 0
    
    def load_data(self, path='./data/ds_salaries.csv'):
        self.df = pd.read_csv(path)
        self.df = self.__drop__(self.df)
        return self.df
    
    def plot_hist(self, data):
        cols = 2
        rows = len(data.columns)//2
        plt.figure(figsize=(4.5*cols, 2.5*rows))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.40)
        for index, col in enumerate(data.columns):
            plt.subplot(rows, cols, index+1)
            plt.hist(data[col])
            plt.title(col.upper())
        plt.show()  
      
    def encode_categorical_data(self, data):
        data["experience_level"] = data["experience_level"].replace({'EX': 4,'SE': 3, 'MI': 2, 'EN': 1})
        data["employment_type"] = data["employment_type"].replace({'FT': 4,'PT': 3, 'CT': 2, 'FL': 1})
        data["work_year"] = data["work_year"].replace({2023: 4, 2022: 3, 2021: 2, 2020: 1})
        data["company_size"] = data["company_size"].replace({'L': 3, 'M': 2, 'S': 1})
        return data

    def normalize_data(self, data):
        for col in data.columns:
            data[col] = (data[col] - data[col].min())/(data[col].max() - data[col].min())
        return data
    
    
    
    def get_data_split_for_classification(self, data):
        data.apply(self.__descretize_salary__, axis="columns")
        y = data["salary_in_usd"]
        X = data.drop(["salary_in_usd"], axis=1)
        return train_test_split(X, y, test_size=0.20, random_state=self.seed)
    
    def get_data_split(self, data):
        y = data["salary_in_usd"]
        X = data.drop(["salary_in_usd"], axis=1)
        return train_test_split(X, y, test_size=0.20, random_state=self.seed)
    
    def oversample(self, X_train, y_train):
        print(X_train.shape, y_train.shape)
        X = X_train.to_numpy()
        y = y_train.to_numpy()
        sm = SMOTE(random_state=self.seed)
        return sm.fit_resample(X, y)
    
