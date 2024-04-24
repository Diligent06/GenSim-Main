import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import os
import re
import itertools
def cos_similarity(embed_j, embed_k):
    return np.dot(embed_j, embed_k) / \
        (np.linalg.norm(embed_j) * np.linalg.norm(embed_k))
def euclidean_dis(embed_j, embed_k):
    return np.sqrt(np.sum(np.power(embed_j - embed_k, 2)))
def corrcoef(embed_j, embed_k):
    return np.corrcoef(embed_j, embed_k)[0, 1]
def manhattan_dis(embed_j, embed_k):
    return np.sum(np.fabs(embed_j - embed_k))
def embedding(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embeddings = []

    for text in texts:
        token = tokenizer(text, return_tensors='pt')
        outputs = model(**token)
        embeddings.append((outputs.last_hidden_state[:, 0, :]).detach().numpy())
    embeddings = np.array(embeddings)
    embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[2])
    return embeddings
def get_trial_error_info(folder):
    errors = []
    for filename in os.listdir(folder):
        if filename.endswith('_error.txt'):
            with open(folder + '/' + filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                last_line = lines[-1]
                errors.append(last_line)
            f.close()
    return errors
def get_trial_task_info_no_desc(folder):
    names = []
    lang_temps = []

    pattern2 = r'self.lang_template = "(.*?)"'
    files = os.listdir(folder)
    files = sorted(files)
    for filename in files:
        if filename.endswith('_code_output.txt'):
            with open(folder + '/' + filename, 'r', encoding='utf-8') as f:
                text = f.read()
                matches = re.findall(pattern2, text, re.DOTALL)
                if len(matches) != 0:
                	lang_temps.append(matches[0])
                else:
                	lang_temps.append('None')
            f.close()
    return names, lang_temps
def get_trial_task_info(folder, pattern_type):
    names = []
    descriptions = []
    lang_temps = []

    pattern1 = r'>>> Answer: \n{(.*?)}\n\n================= Code Generation!\n\n>>> Prompt:'
    if pattern_type == 1:
        pattern1 = r'>>> Answer: \n{(.*?)}\n\n================= Error Book Preview!\n\n>>> Prompt:'
    elif pattern_type == 2:
        pattern1 = r'>>> Answer: \n{(.*?)}\n\n================= API Preview!\n\n>>> Prompt:'
    pattern2 = r'self.lang_template = "(.*?)"'
    files = os.listdir(folder)
    files = sorted(files)
    for filename in files:
        if filename.endswith('_full_output.txt'):
            with open(folder + '/' + filename, 'r', encoding='utf-8') as f:
                taskname = filename[:-16]
                names.append(taskname)
                text = f.read()
                matches = re.findall(pattern1, text, re.DOTALL)
                descriptions.append('{' + matches[-1] + '\n}')
            f.close()
        elif filename.endswith('_code_output.txt'):
            with open(folder + '/' + filename, 'r', encoding='utf-8') as f:
                text = f.read()
                matches = re.findall(pattern2, text, re.DOTALL)
                if len(matches) != 0:
                	lang_temps.append(matches[0])
                else:
                	lang_temps.append('None')
            f.close()
    return names, descriptions, lang_temps
def write_from_list(text_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for text in text_list:
            if text.endswith('\n'):
                f.write(text)
            else:
                f.write(text + '\n')
        f.close()

def cal_diversity_from_npy(npy_path, similar_method):
    lang_embeddings = np.load(npy_path)
    score = 0
    pairs = list(itertools.combinations(range(lang_embeddings.shape[0]), 2))
    if similar_method == 'cos_similar':
        for j, k in pairs:
            score += 1. - cos_similarity(lang_embeddings[j,:], lang_embeddings[k,:])
    elif similar_method == 'euclidean_dis':
        for j, k in pairs:
            score += 1. - euclidean_dis(lang_embeddings[j,:], lang_embeddings[k,:])
    elif similar_method == 'manhattan_dis':
        for j, k in pairs:
            score += 1. - manhattan_dis(lang_embeddings[j,:], lang_embeddings[k,:])
    else:
        for j, k in pairs:
            score += 1. - corrcoef(lang_embeddings[j,:], lang_embeddings[k,:])
    return score / len(pairs)
def cal_diversity(lang_embeddings, similar_method):
    score = 0
    pairs = list(itertools.combinations(range(lang_embeddings.shape[0]), 2))
    if similar_method == 'cos_similar':
        for j, k in pairs:
            score += 1. - cos_similarity(lang_embeddings[j,:], lang_embeddings[k,:])
    elif similar_method == 'euclidean_dis':
        for j, k in pairs:
            score += 1. - euclidean_dis(lang_embeddings[j,:], lang_embeddings[k,:])
    elif similar_method == 'manhattan_dis':
        for j, k in pairs:
            score += 1. - manhattan_dis(lang_embeddings[j,:], lang_embeddings[k,:])
    else:
        for j, k in pairs:
            score += 1. - corrcoef(lang_embeddings[j,:], lang_embeddings[k,:])
    return score / len(pairs)

def clear_messages():
    global existing_messages
    existing_messages = []