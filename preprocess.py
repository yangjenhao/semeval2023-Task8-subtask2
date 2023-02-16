# -*- encoding: utf-8 -*-
"""
@File   :   preprocess.py
@Desc   :   預處理數據集（轉換為simple transfomer可使用的形式)
"""

import json
import pandas as pd
import re
import string
# import nltk.stem


def prepocess_pico(path):
    df_train = pd.read_csv(path)
    df_train = df_train[['sentence_id','words','labels']]
    df_train = df_train.dropna(axis=0,how='any')
    check_for_nan = df_train.isnull().any().any()
    print('check_for_if_nan = ', check_for_nan)
    return df_train

if __name__ == '__main__':
    df = prepocess_pico('data/test_data.csv')
    print(df)
    print('==================== Finished ====================')