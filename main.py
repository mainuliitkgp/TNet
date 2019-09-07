# -*- coding: utf-8 -*-
import argparse
import math
import time
import os
from layer import TNet
from utils import *
from nn_utils import *
from evals import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TNet settings')
    parser.add_argument("-ds_name", type=str, default="14semeval_rest", help="dataset name")
    parser.add_argument("-n_filter", type=int, default=50, help="number of convolutional filters")
    parser.add_argument("-bs", type=int, default=64, help="batch size")
    parser.add_argument("-dim_w", type=int, default=300, help="dimension of word embeddings")
    parser.add_argument("-dim_e", type=int, default=25, help="dimension of episode")
    parser.add_argument("-dim_func", type=int, default=10, help="dimension of functional embeddings")
    parser.add_argument("-dim_p", type=int, default=30, help="dimension of position embeddings")
    parser.add_argument("-dropout_rate", type=float, default=0.3, help="dropout rate for sentimental features")
    parser.add_argument("-dim_h", type=int, default=50, help="dimension of hidden state")
    parser.add_argument("-rnn_type", type=str, default='LSTM', help="type of recurrent unit")
    parser.add_argument("-n_epoch", type=int, default=100, help="number of training epoch")
    parser.add_argument("-dim_y", type=int, default=3, help="dimension of label space")
    parser.add_argument("-lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("-lambda", type=float, default=1e-4, help="L2 coefficient")
    parser.add_argument("-did", type=int, default=2, help="gpu device id")
    parser.add_argument("-connection_type", type=str, default="AS", help="connection type, only AS and LF are valid")

    args = parser.parse_args()

    args.kernels = [3]

    if args.ds_name == '14semeval_rest':
        args.bs = 25

    dataset, embeddings, embeddings_func, n_train, n_test = build_dataset(ds_name=args.ds_name, bs=args.bs,
                                                                          dim_w=args.dim_w, dim_func=args.dim_func)

    # update the size of the used word embeddings
    args.dim_w = len(embeddings[1])
    print(args)
    args.embeddings = embeddings
    # args.embeddings_func = embeddings_func
    args.sent_len = len(dataset[0][0]['wids'])
    args.target_len = len(dataset[0][0]['tids'])
    print("sent length:", args.sent_len)
    print("target length:", args.target_len)
    print("length of padded training set:", len(dataset[0]))

    n_train_batches = math.ceil(n_train / args.bs)
    # print("n batches of training set:", n_train_batches)
    n_test_batches = math.ceil(n_test / args.bs)
    train_set, test_set = dataset

    cur_model_name = 'TNet-%s' % args.connection_type
    print("Current model name:", cur_model_name)
    model = TNet(args=args)
    # print model summary
    print(model)

    result_strings = []

    for i in range(1, args.n_epoch + 1):
        # ---------------training----------------
        print("In epoch %s/%s:" % (i, args.n_epoch))
        np.random.shuffle(train_set)
        train_y_pred, train_y_gold = [], []
        train_losses = []
        for j in range(n_train_batches):
            # print("In %s-th batch" % j)
            train_x, train_xt, train_y, train_pw = get_batch_input(dataset=train_set, bs=args.bs, idx=j)
            # print(train_x.shape)
            # print(train_xt.shape)
            # print("Before training")
            y_pred, y_gold, loss, _ = model.train(train_x, train_xt, train_y, train_pw, np.int32(1))
            # print("After training...")
            train_y_pred.extend(y_pred)
            train_y_gold.extend(y_gold)
            train_losses.append(loss)
        acc, f, _, _ = evaluate(pred=train_y_pred, gold=train_y_gold)
        print("\ttrain loss: %.4f, train acc: %.4f, train f1: %.4f" % (sum(train_losses), acc, f))

        # ---------------prediction----------------
        test_y_pred, test_y_gold = [], []
        test_feat_maps = []
        for j in range(n_test_batches):
            test_x, test_xt, test_y, test_pw = get_batch_input(dataset=test_set, bs=args.bs, idx=j)
            beg = time.time()
            y_pred, y_gold, loss, batch_feat_map = model.test(test_x, test_xt, test_y, test_pw, np.int32(0))
            end = time.time()
            # print("Model: %s, bs: %s, time: %s" % (cur_model_name, args.bs, end - beg))
            test_y_pred.extend(y_pred)
            test_y_gold.extend(y_gold)
            # test_feat_maps.extend(batch_feat_map)
        acc, f, _, _ = evaluate(pred=test_y_pred[:n_test], gold=test_y_gold[:n_test])
        print("\tperformance of prediction: acc: %.4f, f1: %.4f" % (acc, f))
        result_strings.append("In Epoch %s: accuracy: %.2f, macro-f1: %.2f\n" % (i, acc * 100, f * 100))

    result_logs = ['-------------------------------------------------------\n']
    params_string = str(args)
    result_logs.append("Running model: %s\n" % cur_model_name)
    result_logs.append(params_string + "\n")
    result_logs.extend(result_strings)
    result_logs.append('-------------------------------------------------------\n')
    if not os.path.exists('./log'):
        os.mkdir('log')
    with open("./log/%s_%s.txt" % (cur_model_name, args.ds_name), 'a') as fp:
        fp.writelines(result_logs)