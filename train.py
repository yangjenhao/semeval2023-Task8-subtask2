# -*- encoding: utf-8 -*-
'''
@File   :   train.py
@Desc   :   訓練NERmodel
'''
from config import set_args
from preprocess import prepocess_pico
from simpletransformers.ner import NERModel
import sklearn
import logging
import torch
import sys
print('sys.getrecursionlimit = ', sys.getrecursionlimit())
FORMAT = '%(asctime)s %(levelname)s: %(message)s'
import warnings
warnings.filterwarnings('ignore')

##########? labels_list
labels_list = ['B-intervention', 'O', 'B-outcome',  "B-population"]

def train_model():
    ##########? 訓練參數
    
    args = set_args()
    logging.basicConfig(level=logging.INFO, filename='logs/train.log', filemode='w', format=FORMAT)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    ##########? 讀取pico數據
    train_df = prepocess_pico(args.pico_train_path)
    dev_df = prepocess_pico(args.pico_dev_path)
    print(labels_list)
    ##########? 創建命名實體辨別任務NER model
    cuda_available = torch.cuda.is_available()
    model = NERModel(args.model_type, args.model_name, use_cuda=cuda_available, labels=labels_list, args=vars(args))
    model.save_model(model=model.model)  # 可以將預訓練模型下載到output_dir

    ##########? 訓練模型，並在訓練時評估
    model.train_model(train_df, eval_data=dev_df)
    

if __name__ == '__main__':
    train_model()
    print('==================== Training Finished ====================')
