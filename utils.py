import os
import pandas as pd
import numpy as np

def save_results_in_csv(y_pred):
    '''
    Saves the predictions of test lung images in the provided csv file
    :param y_pred: numpy array containing the boolean predictions
    '''
    fname = os.path.join(os.path.abspath(__file__), '..', 'data', 'CPSC340_Q2_SUBMISSION.csv')
    y_pred = y_pred.astype(bool)

    df = pd.DataFrame(columns=['Id', 'Predicted'])
    df['Id'] = np.arange(0, y_pred.shape[0])
    df['Predicted'] = y_pred
    df.to_csv(fname, index=False)