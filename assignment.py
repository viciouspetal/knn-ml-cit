import numpy as np
import pandas as pd


class Assignment:
    path_to_cancer = './dataset/cancer'
    path_to_cancer_training = path_to_cancer + '/trainingData2.csv'
    cancer_dataset_column_headers = ['bi_rads', 'age', 'shape', 'margin', 'density', 'severity']

    def calculate_distances(self, a, b):
        dist = np.linalg.norm(a - b)
        return dist

    def load_data(self, path):
        df = pd.read_csv(path, names=self.cancer_dataset_column_headers, header=None)
        print(df.head())
        return df

    def main(self):
        # first need to load the training dataset
        df = self.load_data(self.path_to_cancer_training)

        #converting pandas dataframe to numpy array
        numpyArray = df.values
        #distances = self.calculate_distances(numpyArray) -> need to refactor calc dist from basic to one taking in above array


if __name__ == '__main__':
    subject = Assignment()
    subject.main()
