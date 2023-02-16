# -*- encoding: utf-8 -*-
"""
@File   :   config.py
@Desc   :   模型訓練超參數以及模型儲存位置，等其他參數：
"""
import argparse
# 
def data_config(parser):
    ##########? datasets ##########
    model_name = '#roberta4e-5'
    fold_name ='1'
    parser.add_argument("--pico_train_path", type=str, default="data/_SemEval2023_/" + str(fold_name) + "fold_train.csv",
                        help="pico的資料")
    parser.add_argument("--pico_dev_path", type=str, default="data/_SemEval2023_/" + str(fold_name) + "fold_test.csv",
                        help="pico的資料")
    parser.add_argument("--pico_test_path", type=str, default='data/semEval-TEST/TESTDATA.csv',
                        help="pico的資料")
    ##########? data_config ##########
    parser.add_argument("--reprocess_input_data", type=bool, default=True,
                        help="如果為True，則即使cache_dir中存在輸入數據的緩存文件，也將重新處理輸入數據")
    parser.add_argument("--overwrite_output_dir", type=bool, default=True,
                        help="如果為True，則訓練後的模型將保存到ouput_dir，並且覆蓋同一目錄中現有的已保存模型")
    parser.add_argument("--use_cached_eval_features", type=bool, default=True,
                        help="訓練期間的評估使用緩存特徵，將此設置為False將導致在每個評估步驟中重新計算特徵")
    parser.add_argument("--output_dir", type=str, default="/workplace/jhyang/semEavlTask8_2023/simpleTransformer/#2paper_model/" + str(model_name) + "/Fold" + str(fold_name),
                        help="存儲所有輸出，包括模型checkpoints和評估結果")
    parser.add_argument("--best_model_dir", type=str, default="/workplace/jhyang/semEavlTask8_2023/simpleTransformer/#2paper_model/" + str(model_name) + "/Fold" + str(fold_name) + "/best_model/",
                        help="保存評估過程中的最好模型")
    parser.add_argument("--best_model_dir", type=str, default=False,
                        help="保存評估過程中的最好模型")
    return parser

def model_config(parser):
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="模型支持的最大序列長度")
    parser.add_argument("--model_type", type=str, default="roberta",
                        help="模型類型 bert/roberta")
    parser.add_argument("--model_name", type=str, default="roberta-large",
                        help="選擇使用哪個預訓練模型")
    parser.add_argument("--manual_seed", type=int, default=777,
                        help="為了產生可重現的結果，需要設置隨機種子")
    return parser

def train_config(parser):
    parser.add_argument("--evaluate_during_training", type=bool, default=True,
                        help="設置為True以在訓練模型時執行評估，確保評估數據已傳遞到訓練方法")
    parser.add_argument("--num_train_epochs", type=int, default=20,
                        help="模型訓練迭代數")
    parser.add_argument("--evaluate_during_training_steps", type=int, default=210,
                        help="在每個指定的step上執行評估，checkpoint和評估結果將被保存")
    parser.add_argument("--save_eval_checkpoints", type=bool, default=False)
    parser.add_argument("--save_model_every_epoch", type=bool, default=True, help="每次epoch保存模型")
    parser.add_argument("--save_steps", type=int, default=-1, help='Save a model checkpoint at every specified number of steps. Set to -1 to disable.')
    parser.add_argument("--n_gpu", type=int, default=1,help="訓練時使用的GPU個數")
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    # learning_rate
    parser.add_argument("--learning_rate", type=float, default=4e-5, help="學習速率設置")
    return parser


def set_args():
    parser = argparse.ArgumentParser()
    parser = data_config(parser)
    parser = model_config(parser)
    parser = train_config(parser)

    args, unknown = parser.parse_known_args()
    return args
