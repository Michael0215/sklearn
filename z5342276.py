import pandas as pd, warnings, os, sys
warnings.filterwarnings('ignore')

from copy import deepcopy
from scipy.stats import pearsonr

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# model or select features
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report, mean_squared_error

## regression selector
from sklearn.feature_selection import f_regression, VarianceThreshold
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge

## classification selector
from sklearn.feature_selection import f_classif

## regression
from sklearn.ensemble import GradientBoostingRegressor

## classification
from sklearn.naive_bayes import GaussianNB

# Data Preprocessing
def preprocessing_data(data):
    # replacemnent of NaN
    for col in data.columns:
        if str(data[col].dtypes) == 'object':
            data[col].fillna('unknown', inplace=True)
        else:
            data[col].fillna(data[col].mean(), inplace=True)

    #Lable Encoding
    for i in range(data.shape[1]):
        if type(data.iat[0,i]) == str:
            data[list(data.columns)[i]] = LabelEncoder().fit_transform(data[list(data.columns)[i]].astype('str'))
    preprocessing_data = data
    return preprocessing_data

########################################################################
def Regression(train_set, test_set, model):
    train_set = preprocessing_data(train_set)
    low_var_colunms = train_set.columns[VarianceThreshold(threshold=0.025).fit(train_set).get_support()]
    low_var_set = train_set[low_var_colunms]
    train_set_Y = low_var_set['AMT_INCOME_TOTAL']
    train_set_X = low_var_set.drop('AMT_INCOME_TOTAL', axis=1)
    feature_columns = train_set_X.columns[SelectKBest(score_func=f_regression, k=32).fit(train_set_X, train_set_Y).get_support()]
    # standardize
    standardize = MinMaxScaler(feature_range=(0,1), copy=True, clip=False)
    train_set_X = standardize.fit(train_set_X[feature_columns].values).transform(train_set_X[feature_columns].values)
    # generate model
    model.fit(train_set_X, train_set_Y)
    test_set = preprocessing_data(test_set)
    test_set_Y = test_set['AMT_INCOME_TOTAL']
    test_set_X = standardize.transform(test_set[feature_columns].values)
    predict_test_Y = model.predict(test_set_X)

    part1_output = test_set['SK_ID_CURR'].to_frame()
    part1_output['predicted_income'] = deepcopy(predict_test_Y).tolist()
    part1_output = part1_output.sort_values(by=['SK_ID_CURR'])
    os.makedirs('./', exist_ok=True)
    part1_output.to_csv('./z5342276.PART1.output.csv', index=False)
    part1_summary = pd.DataFrame({'zid': ['z5342276'],\
                                  'MSE': [mean_squared_error(test_set_Y, predict_test_Y)],\
                                  'correlation': [pearsonr(test_set_Y, predict_test_Y)[0]]})
    os.makedirs('./', exist_ok=True)
    part1_summary.to_csv('./z5342276.PART1.summary.csv', index=False)
    # print('MSE=' + str(mean_squared_error(test_set_Y, predict_test_Y)) + \
    #       ', correlation=' + str(pearsonr(test_set_Y, predict_test_Y)[0]))
################################################################################
def Classification(train_set, test_set, model):
    train_set = preprocessing_data(train_set)
    low_var_colunms = train_set.columns[VarianceThreshold(threshold=0.025).fit(train_set).get_support()]
    low_var_set = train_set[low_var_colunms]
    train_set_Y = low_var_set['TARGET']
    train_set_X = low_var_set.drop('TARGET', axis=1)
    feature_columns = train_set_X.columns[SelectKBest(score_func=f_classif, k=8).fit(train_set_X, train_set_Y).get_support()]
    # standardize retrieve from: https://www.datatrigger.org/post/scaling/
    standardize = MinMaxScaler(feature_range=(0, 1), copy=True, clip=False)
    train_set_X = standardize.fit(train_set_X[feature_columns].values).transform(train_set_X[feature_columns].values)
    # generate model
    model.fit(train_set_X, train_set_Y)
    test_set = preprocessing_data(test_set)
    test_set_Y = test_set['TARGET']
    test_set_X = standardize.transform(test_set[feature_columns].values)
    predict_test_Y = model.predict(test_set_X)
    part2_output = test_set['SK_ID_CURR'].to_frame()
    part2_output['predicted_target'] = deepcopy(predict_test_Y).tolist()
    part2_output = part2_output.sort_values(by=['SK_ID_CURR'])
    os.makedirs('./', exist_ok=True)
    part2_output.to_csv('./z5342276.PART2.output.csv', index=False)
    report = classification_report(test_set_Y, predict_test_Y, output_dict=True)
    part2_summary = pd.DataFrame({'zid': ['z5342276'],\
                                  'average_precision': [report['macro avg']['precision']],\
                                  'average_recall': [report['macro avg']['recall']], \
                                  'accuracy': [report['accuracy']]})
    os.makedirs('./', exist_ok=True)
    part2_summary.to_csv('./z5342276.PART2.summary.csv', index=False)
    # print('average_precision=' + str(report['macro avg']['precision'])\
    #        + ', average_recall=' + str(report['macro avg']['recall'])\
    #        + ', accuracy=' + str(report['accuracy']))
################################################################################
def main(training_csv, test_csv):
    train_data = pd.read_csv(training_csv)
    test_data = pd.read_csv(test_csv)
    Regression(train_data, test_data, SGDRegressor())
    Classification(train_data, test_data, GaussianNB(priors=None, var_smoothing=1e-9))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

