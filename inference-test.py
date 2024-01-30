import torch
import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import datasets
from datasets import load_dataset, load_from_disk


MODEL_PATH="Model/"
CACHE_DIR= "datasets/"

# MODEL_NAME = "models--deepset--roberta-base-squad2"
# MODEL_POS = MODEL_PATH+MODEL_NAME
# # MODEL_POS = "Model/models--deepset--roberta-base-squad2/snapshots/e84d19c1ab20d7a5c15407f6954cef5c25d7a261/"
# print(MODEL_POS)
#加载模型和分词器


dataset = load_dataset("ms_marco",'v1.1',cache_dir = CACHE_DIR)
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2",cache_dir=MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")


# 定义问题和上下文
question = "does human hair stop squirrels"
context = "Spread some human hair around your vegetable and flower gardens. This will scare the squirrels away because humans are predators of squirrels. It is better if the hair hasn't been washed so the squirrels will easily pick up the human scent."
# 使用分词器处理文本
inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")

# 使用模型进行预测
output = model(**inputs)

# 获取答案的起始和结束位置
answer_start = torch.argmax(output.start_logits)
answer_end = torch.argmax(output.end_logits) + 1

# 将 token ID 转换回文本
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

print(answer)