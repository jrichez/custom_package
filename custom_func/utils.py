# Preprocessing
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Metrics
from sklearn.metrics import mean_squared_error

def encoder(dataset, encoder=LabelEncoder()):
    """Encode the numerical features of a dataset
    
    :param dataset: dataframe
    :return: return a copy of the initial dataset 

    >>> encoder(df)
    df
    """
    var_obj = dataset.select_dtypes(include='object').columns
    dataset_cop = dataset.copy()
    le = encoder
    for i in var_obj:
        dataset_cop[i] = le.fit_transform(dataset_cop[i]) 
    return dataset_cop

def get_data(dataset):
    """Split the train set and the test set and divide the train set between dependant variable and independant features
    
    :param dataset: dataframe
    :return: the dependant variable, the independant features of the train set and the test set

    >>> get_data(df)
    X, y, df_test
    """
    dataset = encoder(dataset)
    df_test = dataset[dataset.index > train_id].copy()
    X = dataset.loc[set(dataset.index) - set(df_test.index)].copy()
    y = X.pop('SalePrice')
    return X, y, df_test

def split(X, y, seed=6):
    """Split the train set and the validation set
    
    :param X: dependant variable
    :param y: independant features
    :return: the train set and the validation set

    >>> split(df)
    X_train, X_valid, y_valid, y_test
    """
    X_train, X_valid, y_valid, y_test = train_test_split(X, y, random_state=seed)
    return X_train, X_valid, y_valid, y_test

def scaler(X_train, X_valid, scaler_func, df_test=None):
    """Scale the data
    
    :param X_train: train set 
    :param X_valid: validation set 
    :param scaler_func: the scaler function to use 
    :param df_test: test set, dy default None
    :return: the train set and the validation set

    >>> scaler(X_train, X_valid, StandardScaler(), df_test)
    X_train, X_valid, df_test
    """
    scal = scaler_func
    X_train = scal.fit_transform(X_train)
    X_test = scal.transform(X_test)
    if df_test is not None:
        df_test.drop('SalePrice', axis=1, inplace=True)
        df_test = scal.transform(df_test)
    return X_train, X_test, df_test

def get_score(dataset, estimator, scaler_func=StandardScaler()):
    """Calcul the rmse of a model
    
    :param dataset: dataframe
    :param estimator: the algorithm to use 
    :param scaler_func: the scaler function to use, by default StandardScaler
    :return: the rmse score 

    >>> get_score(df, CatboostRegressor(), StandardScaler())
    0.15
    """   
    dataset = encoder(dataset)
    X, y, _ = get_data(dataset)
    X_train, X_test, y_train, y_test = split(X, y)
    X_train, X_test, _ = scaler(X_train, X_test, scaler_func)
    model = estimator.fit(X_train, y_train)
    pred = model.predict(X_test)
    # we use log to be on the same scale than the leaderboard
    score = mean_squared_error(np.log(y_test), np.log(pred), squared=False)
    return score

def get_sub(dataset, estimator, scaler_func=StandardScaler()):
    """Provide the submission file with the predictions
    
    :param dataset: dataframe
    :param estimator: the algorithm to use 
    :param scaler_func: the scaler function to use, by default StandardScaler
    :return: the submission file

    >>> get_sub(df, CatboostRegressor(), StandardScaler())
    sub
    """ 
    X, y, df_test = get_data(dataset)
    X_train, X_test, y_train, y_test = split(X, y)
    X_train, X_test, df_test = scaler(X_train, X_test, scaler_func, df_test)
    model.fit(X_train, y_train)
    pred = model.predict(df_test)
    sub['SalePrice'] = pred
    return sub

