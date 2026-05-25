from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import set_seed
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import os
os.environ['HF_TOKEN'] = "<hf_token>"


seed = 42
set_seed(seed)
system_prompt = "You are a patient that has gone to do an interview with a psychologist. The psychologist will ask you a series of questions and you will answer them in a natural way:\n"
user_prompt = "### Input:\n{question}\n\n### Expected Response:\n{answer}"

def apply_prompt(example):
    example["text"] = (
        system_prompt
        + user_prompt.format(question=example["question"], answer=example["answer"])
    )
    return example
#Loading data
def create_dataset(partition:str)->pd.DataFrame:
    dataset_pt = load_dataset('json', data_files=f'../../data/ordered_PT_{partition}_dataset.json')['train']
    dataset_hc = load_dataset('json', data_files=f'../../data/ordered_healthy_{partition}_dataset.json')['train']
    dataset_pt = dataset_pt.map(apply_prompt)
    dataset_hc = dataset_hc.map(apply_prompt)
    dataset_pt = dataset_pt.to_pandas()
    dataset_hc = dataset_hc.to_pandas()
    dataset_pt['scz'] = 1
    dataset_hc['scz'] = 0
    return pd.concat([dataset_hc, dataset_pt], ignore_index=True)
df_train = create_dataset('train')
df_val = create_dataset('eval')
df_test = create_dataset('test')

#Loading models
model_name_pt = "PabloCano1/ordered-PT-gemma3-4b-fine-tuned"
model_name_hc = "PabloCano1/ordered-HC-gemma3-4b-fine-tuned"

print(f"Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name_pt)
print(f"Loading models")
model_hc = AutoModelForCausalLM.from_pretrained(
    model_name_hc,
    device_map="cuda",
    dtype="bfloat16",
  attn_implementation="flash_attention_2",
)
model_pt = AutoModelForCausalLM.from_pretrained(
    model_name_pt,
    device_map="cuda",
    dtype="bfloat16",
  attn_implementation="flash_attention_2",
)
model_hc.eval()
model_pt.eval()

def calc_perplexity(df):
    perplexities_hc = []
    perplexities_pt = []
    with torch.no_grad():
        for text in tqdm(df['text'], desc="Calculando perplexities"):
            enc = tokenizer(text, return_tensors="pt").to(model_hc.device)
            outputs_hc = model_hc(**enc, labels=enc["input_ids"])
            loss_hc = outputs_hc.loss.item()  # loss media por token (cross-entropy)
            ppl_hc = float(np.exp(loss_hc))
            perplexities_hc.append(ppl_hc)
            outputs_pt = model_pt(**enc, labels=enc["input_ids"])
            loss_pt = outputs_pt.loss.item()  # loss media por token (cross-entropy)
            ppl_pt = float(np.exp(loss_pt))
            perplexities_pt.append(ppl_pt)
            
    df['per_pt_model'] = perplexities_pt
    df['per_hc_model'] = perplexities_hc
    df['per_pt_model'] = df['per_pt_model'].round(2)
    df['per_hc_model'] = df['per_hc_model'].round(2)
    return df
df_train = calc_perplexity(df_train)
df_val = calc_perplexity(df_val)
df_test = calc_perplexity(df_test)

df_train.to_csv("../../data/train_text_with_perplexities.csv")
df_val.to_csv("../../data/val_text_with_perplexities.csv")
df_test.to_csv("../../data/test_text_with_perplexities.csv")