import numpy as np
import torch
from transformers.trainer_utils import set_seed
import torch.nn.functional as F
import openai

import time
import random
import json


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

with open('random_contexts_3.json') as json_file:
    random_cont = json.load(json_file)

openai.organization = "insert_organization"
openai.api_key = "insert_your_API_key"

def unbiased_sampling(context, n_samples, t):
    prompt = f"You are ChatGPT, a large language model trained by OpenAI. I want you to answer which word is a plausible continuation to the context '{context}'. I have no specific intent, I just want your guess. Return only the word and nothing else."
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                            messages = [{"role": "system", "content" : prompt}],
                                            temperature = t,
                                            top_p = 1,
                                            n = n_samples,
                                            max_tokens = 5)
    
    words = [completion['message']['content'] for completion in response.choices]
    return words

def diverse_sampling(context, n_samples, t):
    prompt = f"You are ChatGPT, a large language model trained by OpenAI. I want you to answer which {n_samples} words are plausible continuations to the context '{context}'. I have no specific intent, I just want your guess. Return only the words and nothing else."
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                            messages = [{"role": "system", "content" : prompt}],
                                            temperature = t,
                                            top_p = 1,
                                            n = 1,
                                            max_tokens = 250)
    
    words = response['choices'][0]['message']['content']
    return words

def generate(contexts, n_samples):
    for context in contexts:
        generations = {}
        words_1 = unbiased_sampling(context, n_samples, 1)
        time.sleep(random.random()*15)

        words_2 = unbiased_sampling(context, n_samples, 2)
        time.sleep(random.random()*15)

        div_words = diverse_sampling(context, n_samples, 1)

        generations['unbiased_temp_1'] = words_1
        generations['unbiased_temp_2'] = words_2
        generations['diverse'] = div_words

        context.append(generations)
        time.sleep(random.random()*15)
    
    return contexts

def compute_generations(lower_cont, upper_cont):
    contexts = random_cont[lower_cont:upper_cont]
    contexts_with_generations = generate(contexts, n_samples = 40)

    json_string = json.dumps(contexts_with_generations)

    with open('chatgpt_generations-' + str(lower_cont) + '-' + str(upper_cont-1) + '.json', 'w') as outfile:
        outfile.write(json_string)

#set_seed(1)
compute_generations(lower_cont = 162, upper_cont = 166)
