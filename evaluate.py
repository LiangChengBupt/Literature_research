import torch
import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import datasets
from datasets import load_dataset, load_from_disk
import logging
import datetime
log_filename = datetime.datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.txt")
MODEL_PATH="Model/"
CACHE_DIR= "datasets/"
LOG_DIR = "LOGS/"
# level=logging.DEBUG
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    filename=LOG_DIR+log_filename,
                    filemode='w')



dataset = load_dataset("ms_marco",'v1.1',cache_dir = CACHE_DIR)
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2",cache_dir=MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

test_set = dataset["test"]

data ={
    "query_id":[],
    "generated_answer":[],
    "correct_answer":[],
    "query":[],
    "query_type":[],
    "input_passage":[]
}

i = 1
for row_item in test_set:
    selected_item = row_item['passages']['is_selected']
    passage_item = row_item['passages']['passage_text']
    # print('passage item type : ' + str(passage_item))

    passage_string = str()
    for idx in range(len(selected_item)):
        if selected_item[idx] == 1:
            passage_string += passage_item[idx]
    # print(input_string)
    inputs = tokenizer.encode_plus(row_item['query'], passage_string, add_special_tokens=True, return_tensors="pt")
    output = model(**inputs)
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits) + 1
    gen_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    data['query_id'].append(row_item['query_id'])
    data['generated_answer'].append(gen_answer)
    data['correct_answer'].append(row_item['answers'])
    data['query'].append(row_item['query'])
    data['query_type'].append(row_item['query_type'])
    data['input_passage'].append(passage_string)
    logging.debug('idx '+'query_id : ' + str(row_item['query_id']) + 'gen_answer :'+gen_answer)
    i += 1
    if(i > 5):
        break

df = pd.DataFrame(data)
df.to_csv("output.csv",index=True , encoding="utf-8")
