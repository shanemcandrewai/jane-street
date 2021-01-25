# Jane Street Market Prediction
https://www.kaggle.com/c/jane-street-market-prediction
## pandas environment
### [Accelerated operations](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html?highlight=numexpr#accelerated-operations)
    pip install numexpr
    pip install bottleneck
[bottleneck Python version < 3.9](https://pypi.org/project/Bottleneck/)
### ipython
#### Error
      File "/home/shane/dev/venv/3.8/lib/python3.8/site-packages/jedi/api/__init__.py", line 725, in __init__
        super().__init__(code, environment=environment,
    TypeError: __init__() got an unexpected keyword argument 'column'
##### github issue
[Last jedi release (0.18.0) is incompatible with ipython](https://github.com/ipython/ipython/issues/12740)
##### solution
    pip install jedi==0.17.2
### pandas
## Evaluation
    import pandas as pd
    import numpy as np
    tra = pd.read_csv('jane-street-market-prediction/train.csv')
    tra.drop(['resp_1', 'resp_2', 'resp_3', 'resp_4'], axis=1, inplace=True)
    tra.drop(tra[(tra.weight == 0)].index, inplace=True)
    corr.resp.sort_values(ascending=False)
    tra[(tra.feature_27 > 0) & (tra.feature_31 > 0) & (tra.feature_37 < 0) & (tra.feature_17 < 0)]
    tra['action'] = np.where(tra.resp > 0, 1, 0)
    tra['pj'] = tra.weight * tra.resp * tra.action
    pi = tra.groupby(['date']).sum().pj
    t = pi.sum()/((pi**2).sum()**0.5) * (250/pi.count())**0.5
    u = min(max(t, 0), 6) * pi.sum()
    corr = tra.corr()
