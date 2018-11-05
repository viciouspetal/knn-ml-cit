from sklearn import neighbors
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
import common_utils as cu
from sklearn.model_selection import train_test_split


def experiment():
    path_to_folder = './dataset/regression'
    path_to_training_file = path_to_folder + '/trainingData.csv'
    #path_to_test_file = path_to_folder + '/testData.csv'

    rmse_val = []
    df = cu.load_data(path_to_training_file, None)

    train, test = train_test_split(df, test_size=0.3)


    x_train = train.drop(train.columns[12], axis=1)
    y_train = train[12]
    x_test = test.drop(test.columns[12], axis=1)
    y_test = test[12]

    print('Type of x_train {0}, y_train {1}, x_test {2}, y_test {3}'.format(type(x_train), type(y_train), type(x_test), type(y_test)))

    K = 5
    model = neighbors.KNeighborsRegressor(n_neighbors = K)
    n=df.shape[0]

    model.fit(x_train, y_train)  #fit the model
    pred=model.predict(x_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= {0} is: {1}'.format(K, error))
    rss = n*error**2
    print('RSS value for K={0} is: {1}'.format(K, rss))


    print('R^2 coefficient is: {0}'.format())


if __name__ == '__main__':
    experiment()