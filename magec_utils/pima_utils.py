import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import magec_utils as mg


def pima_data(filename=None, cols_to_corr_new_feature=[None]):
    """
    Load PIMA data, impute, scale and train/valid split
    :return:
    """
    if isinstance(cols_to_corr_new_feature, str):
      cols_to_corr_new_feature = [cols_to_corr_new_feature]
    

    def impute(df):
        out = df.copy()
        cols = list(set(df.columns) - {'Outcome', 'Pregnancies'})
        out[cols] = out[cols].replace(0, np.NaN)
        out[cols] = out[cols].fillna(out[cols].mean())
        return out

    filename = 'diabetes.csv' if filename is None else filename
    pima = pd.read_csv(filename)

    create_rand_vars = False
    if cols_to_corr_new_feature and cols_to_corr_new_feature[0] is not None:
      create_rand_vars = True

    if create_rand_vars:
      import random
      def create_good_random_variable(row):
        cond_val = row[cols_to_corr_new_feature[0]]
        for col in cols_to_corr_new_feature[1:]:
          cond_val * row[col]
        return cond_val + ((random.random() - 0.5) * 0.5)

      def create_bad_random_variable(row):
        return random.random()
    
      good_rand_var = pima.apply(create_good_random_variable, axis=1)
      bad_rand_var = pima.apply(create_bad_random_variable, axis=1)
      pima.insert(8, 'GoodRandVar', good_rand_var)
      pima.insert(9, 'BadRandVar', bad_rand_var)

    seed = 7
    np.random.seed(seed)
    if create_rand_vars is True:
      x = pima.iloc[:, 0:10]
      Y = pima.iloc[:, 10]
    else:
      x = pima.iloc[:, 0:8]
      Y = pima.iloc[:, 8]


    x_train, x_validation, Y_train, Y_validation = train_test_split(x, Y, test_size=0.2, random_state=seed)

    x_train = impute(x_train)
    x_validation = impute(x_validation)

    stsc = StandardScaler()
    xst_train = stsc.fit_transform(x_train)
    xst_train = pd.DataFrame(xst_train, index=x_train.index, columns=x_train.columns)
    xst_validation = stsc.transform(x_validation)
    xst_validation = pd.DataFrame(xst_validation, index=x_validation.index, columns=x_validation.columns)

    # Format
    x_validation_p = xst_validation.copy()
    x_validation_p['timepoint'] = 0
    x_validation_p['case'] = np.arange(len(x_validation_p))
    x_validation_p.set_index(['case', 'timepoint'], inplace=True)
    x_validation_p = x_validation_p.sort_index(axis=1)

    y_validation_p = pd.DataFrame(Y_validation.copy())
    y_validation_p['timepoint'] = 0
    y_validation_p['case'] = np.arange(len(x_validation_p))
    y_validation_p.set_index(['case', 'timepoint'], inplace=True)
    y_validation_p = y_validation_p.sort_index(axis=1)

    # Format
    x_train_p = xst_train.copy()
    x_train_p['timepoint'] = 0
    x_train_p['case'] = np.arange(len(x_train_p))
    x_train_p.set_index(['case', 'timepoint'], inplace=True)
    x_train_p = x_train_p.sort_index(axis=1)

    y_train_p = pd.DataFrame(Y_train.copy())
    y_train_p['timepoint'] = 0
    y_train_p['case'] = np.arange(len(y_train_p))
    y_train_p.set_index(['case', 'timepoint'], inplace=True)
    y_train_p = y_train_p.sort_index(axis=1)

    return pima, x_train, x_validation, stsc, x_train_p, x_validation_p, y_train_p, y_validation_p


def pima_models(x_train_p, y_train_p):
    """
    3 ML models for PIMA (scaled) data
    :param x_train_p:
    :param Y_train:
    :return:
    """

    def create_mlp():
        mlp = Sequential()
        mlp.add(Dense(60, input_dim=len(x_train_p.columns), activation='relu'))
        mlp.add(Dropout(0.2))
        mlp.add(Dense(30, input_dim=60, activation='relu'))
        mlp.add(Dropout(0.2))
        mlp.add(Dense(1, activation='sigmoid'))
        mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return mlp

    mlp = KerasClassifier(build_fn=create_mlp, epochs=100, batch_size=64, verbose=0)
    mlp._estimator_type = "classifier"
    mlp.fit(x_train_p, y_train_p)

    rf = RandomForestClassifier(n_estimators=1000)
    rf.fit(x_train_p, y_train_p)
    sigmoidRF = CalibratedClassifierCV(RandomForestClassifier(n_estimators=1000), cv=5, method='sigmoid')
    sigmoidRF.fit(x_train_p, y_train_p)

    # lr = LogisticRegression(C=1.)
    # lr.fit(x_train_p, y_train_p)

    svm = SVC(kernel='linear', probability=True)
    svm.fit(x_train_p, y_train_p)

    # create a dictionary of our models
    estimators = [('rf', sigmoidRF), ('mlp', mlp), ('svm', svm)]
    # create our voting classifier, inputting our models
    ensemble = VotingClassifier(estimators, voting='soft')
    ensemble._estimator_type = "classifier"
    ensemble.fit(x_train_p, y_train_p)

    return {'mlp': mlp, 'rf': sigmoidRF, 'svm': svm, 'ensemble': ensemble}


def plot_stats(dfplot, save=False):
    dfplot = dfplot.set_index('Feature')
    dfplot.plot(kind='bar',
                stacked=True,
                figsize=(10, 6),
                title='MAgEC (best) features by model and policy',
                rot=45)
    if save:
        plt.savefig('pima_magec_stats.png', bbox_inches='tight')
    return


def df_stats(stats, con1, con3):

    def feat_num(feat, stats, model):
        if feat in stats and np.any([model in x[1] for x in stats[feat]]):
            return [x[0] for x in stats[feat] if x[1] == model][0]
        else:
            return 0

    dfplot = pd.DataFrame(columns=['Feature', 'LR', 'RF', 'MLP', 'CON@1', 'CON@3'],
                          data=[['Glucose',
                                 feat_num('Glucose', stats, 'lr'),
                                 feat_num('Glucose', stats, 'rf'),
                                 feat_num('Glucose', stats, 'mlp'),
                                 con1['Glucose'][0] if 'Glucose' in con1 else 0,
                                 con3['Glucose'][0] if 'Glucose' in con3 else 0],
                                ['Insulin',
                                 feat_num('Insulin', stats, 'lr'),
                                 feat_num('Insulin', stats, 'rf'),
                                 feat_num('Insulin', stats, 'mlp'),
                                 con1['Insulin'][0] if 'Insulin' in con1 else 0,
                                 con3['Insulin'][0] if 'Insulin' in con3 else 0],
                                ['BMI',
                                 feat_num('BMI', stats, 'lr'),
                                 feat_num('BMI', stats, 'rf'),
                                 feat_num('BMI', stats, 'mlp'),
                                 con1['BMI'][0] if 'BMI' in con1 else 0,
                                 con3['BMI'][0] if 'BMI' in con3 else 0],
                                ['BloodPressure',
                                 feat_num('BloodPressure', stats, 'lr'),
                                 feat_num('BloodPressure', stats, 'rf'),
                                 feat_num('BloodPressure', stats, 'mlp'),
                                 con1['BloodPressure'][0] if 'BloodPressure' in con1 else 0,
                                 con3['BloodPressure'][0] if 'BloodPressure' in con3 else 0],
                                ['SkinThickness',
                                 feat_num('SkinThickness', stats, 'lr'),
                                 feat_num('SkinThickness', stats, 'rf'),
                                 feat_num('SkinThickness', stats, 'mlp'),
                                 con1['SkinThickness'][0] if 'SkinThickness' in con1 else 0,
                                 con3['SkinThickness'][0] if 'SkinThickness' in con3 else 0],
                                ['not_found',
                                 feat_num('not_found', stats, 'lr'),
                                 feat_num('not_found', stats, 'rf'),
                                 feat_num('not_found', stats, 'mlp'),
                                 con1['not_found'][0] if 'not_found' in con1 else 0,
                                 con3['not_found'][0] if 'not_found' in con3 else 0]])
    return dfplot


def plot_pima_features(df):
  if 'GoodRandVar' in df.columns:
    fig, ax = plt.subplots(4, 2, figsize=(14, 12))
    ax[0, 0] = mg.plot_feature(df, 'BMI', ax[0, 0])
    ax[0, 1] = mg.plot_feature(df, 'BloodPressure', ax[0, 1])
    ax[1, 0] = mg.plot_feature(df, 'DiabetesPedigreeFunction', ax[1, 0])
    ax[1, 1] = mg.plot_feature(df, 'Glucose', ax[1, 1])
    ax[2, 0] = mg.plot_feature(df, 'Insulin', ax[2, 0])
    ax[2, 1] = mg.plot_feature(df, 'SkinThickness', ax[2, 1])
    ax[3, 0] = mg.plot_feature(df, 'GoodRandVar', ax[3, 0])
    ax[3, 1] = mg.plot_feature(df, 'BadRandVar', ax[3, 1])
  else:
    fig, ax = plt.subplots(3, 2, figsize=(14, 12))
    ax[0, 0] = mg.plot_feature(df, 'BMI', ax[0, 0])
    ax[0, 1] = mg.plot_feature(df, 'BloodPressure', ax[0, 1])
    ax[1, 0] = mg.plot_feature(df, 'DiabetesPedigreeFunction', ax[1, 0])
    ax[1, 1] = mg.plot_feature(df, 'Glucose', ax[1, 1])
    ax[2, 0] = mg.plot_feature(df, 'Insulin', ax[2, 0])
    ax[2, 1] = mg.plot_feature(df, 'SkinThickness', ax[2, 1])
    plt.show()
  return ax
