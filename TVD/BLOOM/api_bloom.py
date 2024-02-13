import numpy as np
import torch
from transformers.trainer_utils import set_seed
import torch.nn.functional as F
import requests

import time
import random
import json

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

with open('random_contexts_4.json') as json_file:
    random_cont = json.load(json_file)


API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
headers = {"Authorization": "Bearer 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"}
#Evgenia
#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

#Wilker
#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

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
    

def generate(contexts, n_samples):
    for context in contexts:
        generations = []
        print(context[0])
        for seed in range(n_samples):
            response = query({"inputs": context[0],
                    "parameters": {"top_k": None,
                                "max_new_tokens": 5,
                                "num_return_sequences": 1,
                                'do_sample': True,
                                'return_full_text': False,
                                'num_beams': 1,
                                'seed': seed}})
            generations.append(response[0]['generated_text'])
            time.sleep(2)

        context.append(generations)
    
    return contexts


def compute_generations(lower_cont, upper_cont):
    contexts = random_cont[lower_cont:upper_cont]
    contexts_with_generations = generate(contexts, n_samples = 40)

    json_string = json.dumps(contexts_with_generations)

    with open('bloom_generations-' + str(lower_cont) + '-' + str(upper_cont-1) + '.json', 'w') as outfile:
        outfile.write(json_string)

compute_generations(lower_cont = 150, upper_cont = 170)