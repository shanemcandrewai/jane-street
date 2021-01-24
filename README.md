# Jane Street Market Prediction
https://www.kaggle.com/c/jane-street-market-prediction
## Evaluation
    import pandas as pd
    tra = pd.read_csv('jane-street-market-prediction/train.csv')
    tra = tra.assign(action=1)
    tra = tra.assign(pj = tra.weight * tra.resp * tra.action)
