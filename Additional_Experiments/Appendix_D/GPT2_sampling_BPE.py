#Importing relevant libraries
import numpy as np
import pandas as pd
import torch
import torch.distributions as td
import torch.utils.data as data
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.trainer_utils import set_seed

import os
import re
import json

#Switching to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_provo_corpus(file_name):
    """Reading the raw Provo Corpus dataset and create a dictionary with all useful 
       information we might need from it"""
    predict_norms = pd.read_csv(file_name, sep='\t')
    paragraphs = predict_norms.groupby('Text_ID')['Text'].max()

    paragraphs_provo = {}
    
    for text_id in range(1,56): #iterate over all provo paragraphs
        paragraphs_provo[text_id] = {'text': paragraphs[text_id], 'predictions':{}}
        for word_num in predict_norms[predict_norms['Text_ID'] == text_id]['Word_Number'].unique(): #iterating over all words in each text
            word_dist = predict_norms[(predict_norms['Text_ID'] == text_id) & (predict_norms['Word_Number'] == word_num)]
            unique_human_words = word_dist['Response'].unique() #all human answered words for each word
            unique_word_dist = []
            for word in unique_human_words:
                unique_word_count = sum(word_dist[word_dist['Response'] == word]['Response_Count']) #getting all counts of the unique word and summing them
                unique_word_dist.append((word, unique_word_count))
            
            word_dist_dict = dict(unique_word_dist)
            paragraphs_provo[text_id]['predictions'][str(word_num)] = {}
            paragraphs_provo[text_id]['predictions'][str(word_num)]['human_dist_over_next_word_pred'] = word_dist_dict 
            paragraphs_provo[text_id]['predictions'][str(word_num)]['context_with_cloze_word'] = paragraphs[text_id].split(' ')[:int(word_num)]
            #store the distribution dictionary in a dictionary for each text word (and their position in the text - in case of duplicate words)   

    return paragraphs_provo


class cond_log_probs:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.bos_token = '<|endoftext|>'
        self.pad_token = '<|endoftext|>'
    
    def get_vocab_of_next_word_dist(self):
        self.vocab = self.tokenizer.encoder
        return self.vocab

    def cond_log_prob_given_context_over_bpe_vocab(self, context):
        inputs =  self.tokenizer(f"{self.bos_token} {context}", return_tensors="pt")['input_ids']
        
        with torch.no_grad():
            outputs = self.model(inputs)
            next_token_logits = outputs[0][:, -1, :]

        return next_token_logits

    def cond_log_prob_given_context_of_specific_word(self, word, context):
        """Receives one word and a context and returns the log probability of the word given the context 
        under the model"""
        
        #For the BPE vocabulary, we obtain the conditional log-prob given the context
        cond_log_prob_of_next_bpe_given_context = self.cond_log_prob_given_context_over_bpe_vocab(context)

        #A word is not necessarily a single BPE. Hence, to locate the cond. log prob for the word given
        #the context from the distribution, we need to tokenize the word and assume that the first token is the
        #'word' we are interested in 
        input_ids = self.tokenizer(' ' + str(word))['input_ids'] #adding the space beforehand since we are tokenizing ex. 
        #' now' not 'now (these two are different for the model) since we are looking at 'the next word' - not a starting one
        bpe_ind = input_ids[0]
        
        cond_log_prob_word_given_context = cond_log_prob_of_next_bpe_given_context[0, bpe_ind]
        return(bpe_ind, cond_log_prob_word_given_context.item())

    def cond_log_prob_given_context_of_list_of_words(self, words, context):
        """Receives a list of words and a context and returns a list of log probabilities of the words given
        the context under the model"""

        #For the BPE vocabulary, we obtain the conditional log-prob given the context
        cond_log_prob_of_next_bpe_given_context = self.cond_log_prob_given_context_over_bpe_vocab(context)

        input_ids = self.tokenizer([(' ' + str(x)) for x in words])['input_ids']
        bpe_inds = [item[0] for item in input_ids]

        cond_log_prob_word_given_context = cond_log_prob_of_next_bpe_given_context[0, bpe_inds]

        return(bpe_inds, cond_log_prob_word_given_context.tolist())

def construct_dataset(paragraphs_provo, tokenizer, model, lower_bound, upper_bound):
    """Given the dictionary containing provo information, a model and a tokenizer we produce a dictionary with all useful
    information we need for all contexts in paragraphs (lower bound, upper bound - 1)"""
    dataset = {}
    cond_log_probs_obj = cond_log_probs(model, tokenizer)
    
    vocab_dict = cond_log_probs_obj.get_vocab_of_next_word_dist()
    dataset['vocab_dict'] = vocab_dict

    for text_id in range(lower_bound, upper_bound): #iterate over all provo paragraphs
        print(text_id)
        dataset[text_id] = {}
        for word_num in paragraphs_provo[text_id]['predictions'].keys():
            print(word_num)
            list_context = paragraphs_provo[text_id]['predictions'][word_num]['context_with_cloze_word'][:-1] 
            #we dont take into account the last word (which is the cloze word/one hot label word)
            context = ' '.join(list_context) #the text that makes up the context at each word step
            
            cond_log_prob_bpe_on_cont = cond_log_probs_obj.cond_log_prob_given_context_over_bpe_vocab(context).tolist()[0]

            dataset[text_id][word_num] = {'context' : context,
                                            'model': {'cond_log_prob':cond_log_prob_bpe_on_cont},
                                            'human': {},
                                            'original': {}
                                         }
            #ORIGINAL
            original_label = re.sub(r'[^\w\s]', '', paragraphs_provo[text_id]['predictions'][word_num]['context_with_cloze_word'][-1])
            dataset[text_id][word_num]['original']['pred'] = original_label 
            bpe_ind, cond_log_prob = cond_log_probs_obj.cond_log_prob_given_context_of_specific_word(original_label, context)
            dataset[text_id][word_num]['original']['cond_log_prob'] = cond_log_prob
            dataset[text_id][word_num]['original']['pred'] = original_label 
            dataset[text_id][word_num]['original']['bpe_ind'] = bpe_ind 

            #HUMAN
            human_list = []
            human_words = list(paragraphs_provo[text_id]['predictions'][word_num]['human_dist_over_next_word_pred'].keys())
            bpe_inds, human_cond_log = cond_log_probs_obj.cond_log_prob_given_context_of_list_of_words(human_words, context)
            for i in range(len(human_words)):
                dict_human = {}
                dict_human['pred'] = human_words[i]
                dict_human['cond_log_prob'] = human_cond_log[i]
                dict_human['bpe_ind'] = bpe_inds[i] #we save these since the bpe might not necessarily match the word in 'pred'
                dict_human['count'] = paragraphs_provo[text_id]['predictions'][word_num]['human_dist_over_next_word_pred'][human_words[i]]
                human_list.append(dict_human)
            
            dataset[text_id][word_num]['human'] = human_list
    
    return dataset

def initiate_dataset_construction(input_data, pre_trained_model = "gpt2", lower_bound = 1, upper_bound = 2):
    #Dictionary with useful information for our analysis from Provo Corpus raw
    paragraphs_provo = preprocess_provo_corpus(input_data)
    
    if pre_trained_model == "gpt2":
        #Construct a GPT-2 tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(pre_trained_model)
        tokenizer.pad_token = tokenizer.eos_token

        #The GPT2 Model transformer with a language modeling head on top 
        model = GPT2LMHeadModel.from_pretrained(pre_trained_model) 

    dict_dataset = construct_dataset(paragraphs_provo, tokenizer, model, lower_bound, upper_bound)

    #file_name = os.path.join(output_folder, f"Paragraphs-{lower_bound}-{upper_bound - 1}.json")
    file_name = f"Paragraphs_BPE-{lower_bound}-{upper_bound - 1}.json"

    with open(file_name, 'w') as fp:
       json.dump(dict_dataset, fp)

input_data = os.path.join(os.getcwd(), 'raw_data/Provo_Corpus.tsv') 
initiate_dataset_construction(input_data, pre_trained_model = "gpt2", lower_bound = 1, upper_bound = 14)
