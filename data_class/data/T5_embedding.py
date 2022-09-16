#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import pickle
import researchpy as rp
from operator import itemgetter

from sklearn.utils import compute_sample_weight
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = ByteLevelBPETokenizer(
    "../salesforce_codeT5/CodeT5-main/tokenizer/salesforce/codet5-vocab.json",
    "../salesforce_codeT5/CodeT5-main/tokenizer/salesforce/codet5-merges.txt",
)


def save_model(file_name, model):
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)


def embed_sequence(model, sequence):
    attention_masks = np.ones(len(sequence), dtype=int)
    model = model.cuda()
    out = model(input_ids=torch.tensor(sequence).cuda().to(torch.int64).unsqueeze(0),
                decoder_input_ids=torch.tensor(sequence).cuda().to(torch.int64).unsqueeze(0))
    pooled_embedding = torch.mean(out.encoder_last_hidden_state[0], dim=0)
    return pooled_embedding.cpu().detach().numpy()


def embed_line_by_line(df, model, df_path):
    df['embeded_sequence_sum'] = None
    df['embeded_sequence_avg'] = None
    for i, row in df.iterrows():
        print(i)
        print(row['sample_id'])
        lines = row['method'].split('\n')
        embeded = []
        for line in lines:
            try:
                if len(line) > 0:
                    tokens = tokenizer.encode(line).ids
                    embeded.append(embed_sequence(model, tokens))
            except Exception as e:
                print('Exception')
                print(e)
                print(line)
        df['embeded_sequence_sum'][i] = np.sum(np.asarray(embeded), axis=0)
        df['embeded_sequence_avg'][i] = np.mean(np.asarray(embeded), axis=0)

    pd.to_pickle(df, df_path)


def embed_class(df, model, df_path):
    df['embeded_sequence'] = None
    for i, row in df.iterrows():
        print(i)
        print(row['sample_id'])
        try:
            embedding = embed_sequence(model, tokenizer.encode(row['method']).ids)
            df['embeded_sequence'][i] = embedding
        except Exception as e:
            print("EXCEPTION " + str(i))
            print(e)

    pd.to_pickle(df, df_path)
    print(df.head())


df = pd.read_csv('../data/data_class.csv')
# print(df.head())


df['label'] = np.where(df.severity == 'none', 0, 1)
# print(df.head())


config = T5Config.from_json_file("../salesforce_codeT5/pretrained_models/codet5_small/config.json")

model_small = T5ForConditionalGeneration(config)
model_small.load_state_dict(torch.load("../salesforce_codeT5/pretrained_models/codet5_small/pytorch_model.bin",
                                       map_location=torch.device('cuda')))

embedding_small_path = './T5/df_dc_embeded_by_line_small.pkl'
embed_line_by_line(df, model_small, embedding_small_path)

embedding_small_path = './T5/df_dc_embeded_small.pkl'
embed_class(df, model_small, embedding_small_path)

del model_small


config = T5Config.from_json_file("../salesforce_codeT5/pretrained_models/codet5_base/config.json")
model_base = T5ForConditionalGeneration(config)
model_base.load_state_dict(torch.load("../salesforce_codeT5/pretrained_models/codet5_base/pytorch_model.bin",
                                      map_location=torch.device('cuda')))

embedding_base_path = './T5/df_dc_embeded_by_line_base.pkl'
embed_line_by_line(df, model_base, embedding_base_path)

embedding_base_path = './T5/df_dc_embeded_base.pkl'
embed_class(df, model_base, embedding_base_path)
