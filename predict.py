# -*- encoding: utf-8 -*-
"""
@File   :   test.py
@Desc   :   測試程序
"""
from config import set_args
from simpletransformers.ner import NERModel
from train import labels_list
import numpy as np
from scipy.special import softmax
from preprocess import prepocess_pico
from preprocess import prepocess_pico＿test
import pandas as pd
import logging
FORMAT = '%(asctime)s %(levelname)s: %(message)s'


def prepocess_data(path):
    df_train = pd.read_csv(path)
    df_train = df_train[['sentence_id','words','labels']]
    df_train = df_train.dropna(axis=0,how='any')
    check_for_nan = df_train.isnull().any().any()
    return df_train
def findnumberlist(df): 
    df = pd.read_csv(df)
    df = df.dropna(axis=0,how='any')
    max_sentences = int(df['sentence_id'].max())
    dflist = ['a{}'.format(i) for i in range(0, max_sentences+1)]
    numberlist, wordslist , subreddit_id, post_id = [], [], [], []
    
    w = []
    for i in range(max_sentences+1):
        dflist[i] = df[df['sentence_id'] == i]
        words = dflist[i]['words'].tolist()
        # subreddit = dflist[i]['subreddit_id'].tolist()
        # posti = dflist[i]['post_id'].tolist()
        for i in range(len(words)):
            if i > 1111:
                break
            w.append(words[i])
            wordslist.append(words[i])
            # subreddit_id.append(subreddit[i])
            # post_id.append(posti[i])
        numberlist.append(len(w))
        w = []
    # print(numberlist)
    return numberlist, wordslist, subreddit_id, post_id


if __name__ == '__main__':
    ##########? set args
    args = set_args()
    labels_list = ['B-intervention', 'O', 'B-outcome',  "B-population"]
    model = NERModel('roberta', "#2paper_model/#roberta4e-5/Fold4/checkpoint-1680-epoch-8", labels=labels_list, args=vars(args))
    ##########? Read data:
    # test_df = prepocess_data(args.pico_test_path)
    test_df = prepocess_data('data/_SemEval2023_/4fold_test.csv')
    # dev_df = prepocess_data('data/_SemEval2023_/4fold_test.csv')
    ##########? Read numberlist
    numberlist, wordslist, subreddit_id, post_id = findnumberlist('data/_SemEval2023_/4fold_test.csv')
    true_labels = test_df['labels'].tolist()
    ##########? Predict Data
    result, model_outputs, preds_list = model.eval_model(test_df)
    ####################!!!!!!!!!!!!!!############
    sentence_id , deberta_large = [], []
    li, cur = [], []
    for i in range(len(preds_list)):
        max_labelcount = numberlist[i]
        for j in range(max_labelcount):
            sentence_id.append(i)
            if j >= len(preds_list[i]):
                deberta_large.append('O')
                cur.append('O')
            else:
                deberta_large.append(preds_list[i][j])
                cur.append(preds_list[i][j])
        li.append(len(cur))
        cur = []
    print('Totals test labels = ', len(deberta_large))
    print('Labels是否對上 = ', li == numberlist)
    print(len(wordslist))
    print(len(deberta_large))
    ##########? Predict data2csv
    data = {'words': wordslist, 'true_labels':true_labels, 'labels': deberta_large}
    df = pd.DataFrame(data=data)  
    ##########? Wheather Save?
    df.to_csv('#submit2semeval#/sub2.csv')
    