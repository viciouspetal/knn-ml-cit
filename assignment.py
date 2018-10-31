import numpy as np
import pandas as pd


class Assignment:
    path_to_cancer = './dataset/cancer'
    path_to_cancer_training = path_to_cancer + '/trainingData2.csv'
    path_to_test=path_to_cancer+'/testData2.csv'
    cancer_dataset_column_headers = ['bi_rads', 'age', 'shape', 'margin', 'density', 'severity']

    def calculate_distances(self, A, B):
        dist = np.sqrt(((A - B)**2).sum(-1)) # row wise sum
        #dist = [np.sqrt((A[row, col][0] - B[row, col][0])**2 + (B[row, col][1] -A[row, col][1])**2) for row in range(2) for col in range(2)]
        print(dist)
        return dist

    def load_data(self, path):
        df = pd.read_csv(path, names=self.cancer_dataset_column_headers, header=None)
       # print(df.head())
       # print(df.shape)
        return df

    def main(self):
        # first need to load the training dataset
        df_training = self.load_data(self.path_to_cancer_training)
        df_test = self.load_data(self.path_to_test)

        #converting pandas dataframe to numpy array
        numpyArray = df_training.values
        distances = self.calculate_distances(numpyArray, numpyArray)
       # print(distances)


if __name__ == '__main__':
    subject = Assignment()
    subject.main()
