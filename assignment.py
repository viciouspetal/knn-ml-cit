import numpy as np
import pandas as pd


class Assignment:
    path_to_cancer = './dataset/cancer'
    path_to_cancer_training = path_to_cancer + '/trainingData2.csv'
    cancer_dataset_column_headers = ['bi_rads', 'age', 'shape', 'margin', 'density', 'severity']

    def calculate_distance(self, a, b):
        dist = np.linalg.norm(a - b)
        return dist

    def load_data(self):
        df = pd.read_csv(self.path_to_cancer_training,
                         names=self.cancer_dataset_column_headers, header=None)
        print(df.head())


if __name__ == '__main__':
    subject = Assignment()
    subject.load_data()
