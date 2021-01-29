# Jane Street Market Prediction
https://www.kaggle.com/c/jane-street-market-prediction
## update main Python packages
    python -m pip install -U pip
    pip install -U numpy pandas scikit-learn ipython
## Windows
### Create a [link](https://www.howtogeek.com/howto/16226/complete-guide-to-symbolic-links-symlinks-on-windows-or-linux) to folder on another drive (may need admin privileges)
    D:\kaggle>mklink /J Scripts c:\Users\mcandrs\dev\venv\3.8\Scripts\
## pandas environment
### [Accelerated operations](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html?highlight=numexpr#accelerated-operations)
    pip install numexpr
    pip install bottleneck
[bottleneck Python version < 3.9](https://pypi.org/project/Bottleneck/)
### ipython
#### jedi error
      File "/home/shane/dev/venv/3.8/lib/python3.8/site-packages/jedi/api/__init__.py", line 725, in __init__
        super().__init__(code, environment=environment,
    TypeError: __init__() got an unexpected keyword argument 'column'
##### github issue
[Last jedi release (0.18.0) is incompatible with ipython](https://github.com/ipython/ipython/issues/12740)
##### solution
    pip install jedi==0.17.2
#### Exception [WinError 995] error
    Exception [WinError 995] The I/O operation has been aborted because of either a thread exit or an application request
##### git issue
[WinError 995](https://github.com/ipython/ipython/issues/12049)
##### solution
    pip install --upgrade prompt-toolkit==2.0.10
### pandas
## Preparation
    import pandas as pd
    import numpy as np
    tra = pd.read_csv('jane-street-market-prediction/train.csv')
    tra.drop(['resp_1', 'resp_2', 'resp_3', 'resp_4'], axis=1, inplace=True)
    tra.drop(tra[(tra.weight == 0)].index, inplace=True)
    corr = tra.corr()
    corr.resp.sort_values(ascending=False)
    tra[(tra.feature_27 > 0) & (tra.feature_31 > 0) & (tra.feature_51 < 0) & (tra.feature_50 < 0)]
    tra['action'] = np.where(((tra.feature_27 > 0) & (tra.feature_31 > 0) & (tra.feature_37 < 0) & (tra.feature_17 < 0)), 1, 0)
    tra['pj'] = tra.weight * tra.resp * tra.action
    pi = tra.groupby(['date']).sum().pj
    t = pi.sum()/((pi**2).sum()**0.5) * (250/pi.count())**0.5
    u = min(max(t, 0), 6) * pi.sum()
## Kaggle notebook errors
### [Exception: You can only call `make_env()` once.](https://www.kaggle.com/c/jane-street-market-prediction/discussion/204958)
#### Solution
    janestreet.make_env.__called__ = False
## HistGradientBoostingRegressor
    import pandas as pd
    import numpy as np
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    X = pd.read_csv('jane-street-market-prediction/train.csv')
    w = X['weight']
    y = np.where((X.resp > 0.05) & (X.weight > 0), 1, 0)
    X.drop(['date', 'weight', 'resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp', 'ts_id'], axis=1, inplace=True)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=0.2, random_state=0)
    del X, y, w
    gbc = HistGradientBoostingClassifier().fit(X_train, y_train, w_train)
    gbc.score(X_test, y_test)
    gbc.predict(X_test)
### [Dockerfile](https://github.com/Kaggle/docker-python/blob/master/Dockerfile)
#### b/176817038 avoid upgrade to 0.24 which is causing issues with hep-ml package.
    pip install scikit-learn==0.23.2 && \
## Evaluation
    import janestreet
    env = janestreet.make_env() # initialize the environment
    iter_test = env.iter_test() # an iterator which loops over the test set

    for (test_df, sample_prediction_df) in iter_test:
      sample_prediction_df.action = np.where(((test_df.feature_27 > 0) & (test_df.feature_31 > 0) & (test_df.feature_37 < 0) & (test_df.feature_17 < 0)), 1, 0)
      env.predict(sample_prediction_df)
## Missing values
### Fast CSV read
    import pandas as pd
    import numpy as np
    import datatable as dt
    X = dt.fread('/kaggle/input/jane-street-market-prediction/train.csv', verbose=True).to_pandas()
### Split training set into features X and labels y
    y = np.where((X.resp > 0), 1, 0)
    no_train_cols = ['date', 'resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp', 'ts_id']
    X.drop(no_train_cols, axis=1, inplace=True)
### Count number of NaN
    X.isna().sum().sum()
### Replace NaN with column mean
    from sklearn.impute import SimpleImputer
    X = pd.DataFrame(SimpleImputer().fit_transform(X), columns=X.columns)
#### Linear Discriminant Analysis (dimension reduction)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X = LinearDiscriminantAnalysis().fit_transform(X, y)
