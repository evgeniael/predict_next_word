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
            
            #SHOW WILKER
            # if sum([x[1] for x in unique_word_dist]) != max(word_dist['Total_Response_Count']):
            #     print('sum', sum([x[1] for x in unique_word_dist]), ' total', max(word_dist['Total_Response_Count']),
            #     ' word', word_num, ' text', text_id)
            #     print(unique_word_dist)
            
            word_dist_dict = dict(unique_word_dist)
            paragraphs_provo[text_id]['predictions'][str(word_num)] = {}
            paragraphs_provo[text_id]['predictions'][str(word_num)]['human_dist_over_next_word_pred'] = word_dist_dict 
            paragraphs_provo[text_id]['predictions'][str(word_num)]['context_with_cloze_word'] = paragraphs[text_id].split(' ')[:int(word_num)]
            #store the distribution dictionary in a dictionary for each text word (and their position in the text - in case of duplicate words)   

    return paragraphs_provo

class log_probs:
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        self.bos_token = '<|endoftext|>'
        self.pad_token = '<|endoftext|>'

    def get_log_prob_from_loss(self, sentence):
        """Receives a sentence and produces its probability under the model"""
        inputs =  self.tokenizer(f"{self.bos_token} {sentence}", return_tensors="pt") 
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        log_prob = - loss*( len(inputs["input_ids"][0]) - 1) #multiplying by length since the total loss 
        #was the average of losses, 1/N Sigma(L_CE) and subtracting 1 to not account for the bos_token
        return log_prob.item()

    def get_log_probs_from_prob_dist(self, sentences):
        """Receives a list of sentences and returns a list of tensors with the probabilities of each sentence under the model"""
        
        #We pad to match the length of the longest sentence
        inputs = self.tokenizer(sentences, padding='longest', return_length=True, return_special_tokens_mask=True, 
                                return_tensors="pt")
        
        #The observation per position is shifted with respect to the input 
        obs_ids = torch.roll(inputs['input_ids'], -1, -1) 
    
        with torch.no_grad():
            outputs = self.model(inputs["input_ids"])
            output_logits = outputs.logits
    
        pT = td.Categorical(logits=output_logits)
        tok_log_probs = pT.log_prob(obs_ids)
        zeroes = torch.zeros(tok_log_probs.shape[0],tok_log_probs.shape[1])
    
        #Log probabilities for non-padded tokens
        tok_log_probs_no_pad = torch.where(obs_ids == self.tokenizer.eos_token_id, zeroes, tok_log_probs)
    
        return tok_log_probs_no_pad.sum(-1) #the log_prob of each sentence


class cond_log_probs:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.log_probs_obj = log_probs(self.model, self.tokenizer)
        
    def get_log_prob_context(self, context):
        self.log_prob_context = self.log_probs_obj.get_log_prob_from_loss(context)
        return self.log_prob_context

    def conditional_log_prob_of_word_given_context(self, word, context):
        """Receives one word and a context and returns the log probability of the word given the context under the model"""
        context_and_word = context + ' ' + word
        log_prob_context_and_word = self.log_probs_obj.get_log_prob_from_loss(context_and_word)
        log_prob_conditional_on_word = log_prob_context_and_word - self.log_prob_context
        
        return(log_prob_conditional_on_word)

    def conditional_log_prob_of_word_list_given_context(self, words, context, log_prob_context):
        """Receives a list of words and a context and returns a list of log probabilities of the words given
           the context under the model"""
        sentences = [f'{self.tokenizer.bos_token} {context} {x}' for x in words] #each sentence is the context and one of the words
        log_probs_context_and_words = self.log_probs_obj.get_log_probs_from_prob_dist(sentences)
        log_probs_conditional_on_words = log_probs_context_and_words - log_prob_context

        return([x.item() for x in log_probs_conditional_on_words])



class greedy:
    def __init__(self,  model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.add_tokens = 5
        self.pad_token = 50256
        self.top_k = 0
        self.length_penalty = 0

    def get_greedy_word(self, context):
        """Given the context, we return the greedy output"""
        inputs_ids = self.tokenizer(context, return_tensors="pt")['input_ids']
        len_ids = len(inputs_ids[0])
        inputs_ids.to(device)
        outputs = self.model.generate(inputs_ids, do_sample = False, num_beams = 1, max_length = (len_ids + self.add_tokens), 
            pad_token_id = self.pad_token, top_k= self.top_k, length_penalty = self.length_penalty) #disabling top_k and having no length penalty

        #Greedy output is context + (self.add_tokens generated) tokens - which can be more than one word                
        greedy_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        #The following extracts only the next word from all generated tokens
        generated_without_context = greedy_output[0].replace(context, '')
        generated_without_context_no_punctuation = re.sub(r'[^\w\s]', '', generated_without_context) #removing punctuation
        list_of_words = generated_without_context_no_punctuation.split(' ') #separating the joint string to retrieve words

        if list_of_words == ['']: #if the list is empty
            greedy_word_prediction = 'Failed to generate word'
        else:
            greedy_word_prediction = next(sub for sub in list_of_words if sub) #first non empty word of a list
    
        return greedy_word_prediction
    

class ancestral:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.cond_log_probs_obj = cond_log_probs(self.model, self.tokenizer)

        self.n_samples = 1000
        self.seed = 0
        self.add_tokens = 5
        self.pad_token = 50256
        self.top_k = 0
        self.length_penalty = 0
    
    def get_ancestral_words(self, context):
        """Given a context we return the words that were generated during ancestral sampling for n_samples"""
        inputs_ids = self.tokenizer(context, return_tensors="pt")['input_ids']
        #OR inputs_ids = torch.tile(inputs_ids, [100,1]) parallelizing method
    
        len_ids = len(inputs_ids[0])
        inputs_ids.to(device)
        torch.manual_seed(self.seed) #set seed
        outputs = self.model.generate(inputs_ids, do_sample=True, num_beams = 1, num_return_sequences= self.n_samples, 
                                      max_length= (len_ids + self.add_tokens), pad_token_id = self.pad_token, top_k= self.top_k, 
                                      length_penalty = self.length_penalty)
        
        #Ancestral generation outputs is context + (self.add_tokens generated) tokens - which can be more than one word per sample                
        ancestral_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        generated_without_context = [x.replace(context + ' ', '').replace(context, '').replace('\n', '') for x in ancestral_outputs]
        generated_without_context_no_punctuation = [re.sub(r'[^\w\s]', '', x) for x in generated_without_context] #removing punctuation 
        list_of_words = [x.split(' ') for x in generated_without_context_no_punctuation]  #separating the list of joint strings to retrieve the words in each string
    
        try:
            #we get the first non-empty elements for all the lists in list_of_words, obtaining all the first sampled words
            sampled_word_predictions = [next(sub for sub in x if sub) for x in list_of_words if x ]
        except:
            #we know there is at least one item that failed to generate a word in the sampled items
            sampled_word_predictions = []
            for i in range(len(list_of_words)):
                if not( (list_of_words[i] == ['']) or (list_of_words[i] == ['', '']) or (list_of_words[i] == ['', '', '']) or (list_of_words[i] == ['', '', '', '']) or (list_of_words[i] == ['', '', '', '', ''])):#if one of the generated samples is nothing
                    sampled_word_predictions.append(next(sub for sub in list_of_words[i] if sub))
                else:
                    sampled_word_predictions.append('Failed to generate word')
    
        return(sampled_word_predictions)

    def create_sampled_model_distribution(self, context, log_prob_context): 
        """Given a context we generate n samples using ancestral sampling, obtain the first generated word
        from the generated tokens, count the occurrences of these words and get their log probabilities given 
        the context. We retrun a list of dictionaries containing the words, their counts and their log probabilities"""
        sampled_words = self.get_ancestral_words(context)
    
        #From the list of all next sampled word predictions we get unique entries and their counts
        words, counts = np.unique(sampled_words, return_counts=True)
        log_probs_cond = self.cond_log_probs_obj.conditional_log_prob_of_word_list_given_context(words, context, log_prob_context)
        
        list_samples = []
        for i in range(len(words)):
            dict_samples = {}
            #per word: {pred: string, count: count in ancestreal-sampled data, logprob: log p(pred|context)}
            dict_samples['pred'] = str(words[i])
            dict_samples['count'] = int(counts[i])
            dict_samples['cond_log_prob'] = log_probs_cond[i]
            list_samples.append(dict_samples)
        
        return(list_samples)


class nucleus:
    def __init__(self, top_p, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.cond_log_probs_obj = cond_log_probs(self.model, self.tokenizer)
        self.top_p = top_p

        self.n_samples = 100
        self.seed = 0
        self.add_tokens = 5
        self.pad_token = 50256
        self.top_k = 0
        self.length_penalty = 0

    def get_nucleus_words(self, context):
        """Given a context we return the words that were generated during nucleus sampling for n_samples"""
        inputs_ids = self.tokenizer(context, return_tensors="pt")['input_ids']
        #OR inputs_ids = torch.tile(inputs_ids, [100,1]) parallelizing method
    
        len_ids = len(inputs_ids[0])
        inputs_ids.to(device)
        torch.manual_seed(self.seed) #set seed
        outputs = self.model.generate(inputs_ids, do_sample=True, num_beams = 1, num_return_sequences= self.n_samples, 
                                      top_p = self.top_p, max_length= (len_ids + self.add_tokens), pad_token_id = self.pad_token, 
                                      top_k = self.top_k, length_penalty = self.length_penalty)
        
        #Generation outputs is context + (self.add_tokens generated) tokens - which can be more than one word per sample                
        nucleus_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        generated_without_context = [x.replace(context + ' ', '').replace(context, '').replace('\n', '') for x in nucleus_outputs]
        generated_without_context_no_punctuation = [re.sub(r'[^\w\s]', '', x) for x in generated_without_context] #removing punctuation 
        list_of_words = [x.split(' ') for x in generated_without_context_no_punctuation]  #separating the list of joint strings to retrieve the words in each string
    
        try:
            #we get the first non-empty elements for all the lists in list_of_words, obtaining all the first sampled words
            nucleus_word_predictions = [next(sub for sub in x if sub) for x in list_of_words if x ]
        except:
            #we know there is at least one item that failed to generate a word in the sampled items
            nucleus_word_predictions = []
            for i in range(len(list_of_words)):
                if not( (list_of_words[i] == ['']) or (list_of_words[i] == ['', '']) or (list_of_words[i] == ['', '', '']) or (list_of_words[i] == ['', '', '', '']) or (list_of_words[i] == ['', '', '', '', ''])):#if one of the generated samples is nothing
                    nucleus_word_predictions.append(next(sub for sub in list_of_words[i] if sub))
                else:
                    nucleus_word_predictions.append('Failed to generate word')
    
        return(nucleus_word_predictions)        
    
    def create_nucleus_model_distribution(self, context, log_prob_context): 
        """Given a context we generate n samples using nucleus sampling, obtain the first generated word
        from the generated tokens, count the occurrences of these words and get their log probabilities given 
        the context. We retrun a list of dictionaries containing the words, their counts and their log probabilities"""
        nucleus_words = self.get_nucleus_words(context)
    
        #From the list of all next sampled word predictions we get unique entries and their counts
        words, counts = np.unique(nucleus_words, return_counts=True)
        log_probs_cond = self.cond_log_probs_obj.conditional_log_prob_of_word_list_given_context(words, context, log_prob_context)
        
        list_samples = []
        for i in range(len(words)):
            dict_samples = {}
            #per word: {pred: string, count: count in ancestreal-sampled data, logprob: log p(pred|context)}
            dict_samples['pred'] = str(words[i])
            dict_samples['count'] = int(counts[i])
            dict_samples['cond_log_prob'] = log_probs_cond[i]
            list_samples.append(dict_samples)
        
        return(list_samples)

def construct_dataset(paragraphs_provo, tokenizer, model, lower_bound, upper_bound):
    """Given the dictionary containing provo information, a model and a tokenizer we produce a dictionary with all useful
    information we need for all contexts in paragraphs (lower bound, upper bound - 1)"""
    dataset = {}
    for text_id in range(lower_bound, upper_bound): #iterate over all provo paragraphs
        dataset[text_id] = {}
        for word_num in paragraphs_provo[text_id]['predictions'].keys():
            cond_log_probs_obj = cond_log_probs(model, tokenizer)
            
            list_context = paragraphs_provo[text_id]['predictions'][word_num]['context_with_cloze_word'][:-1] #we dont take into account the last word (which is the cloze word/ one hot label word)
            context = ' '.join(list_context) #the text that makes up the context at each word step
            
            log_prob_context = cond_log_probs_obj.get_log_prob_context(context) #computing its log-probabilty
            
            dataset[text_id][word_num] = {'context' : { 'text':context, 'log_prob':log_prob_context},
                                            'original': {},
                                            'human': {}, 
                                            'greedy': {},
                                            'ancestral_samples' : {},
                                            'nucleus_samples' : { 'top-p': {}}
                                         }
        
            #ORIGINAL
            original_label = paragraphs_provo[text_id]['predictions'][word_num]['context_with_cloze_word'][-1]
            dataset[text_id][word_num]['original']['pred'] = original_label 
            dataset[text_id][word_num]['original']['cond_log_prob'] = cond_log_probs_obj.conditional_log_prob_of_word_given_context(original_label, context)
        
            #HUMAN
            human_list = []
            human_words = list(paragraphs_provo[text_id]['predictions'][word_num]['human_dist_over_next_word_pred'].keys())
            human_cond_log_prob = cond_log_probs_obj.conditional_log_prob_of_word_list_given_context(human_words, context, log_prob_context)
            for i in range(len(human_words)):
                dict_human = {}
                dict_human['pred'] = human_words[i]
                dict_human['cond_log_prob'] = human_cond_log_prob[i]
                dict_human['count'] = paragraphs_provo[text_id]['predictions'][word_num]['human_dist_over_next_word_pred'][human_words[i]]
                human_list.append(dict_human)
            
            dataset[text_id][word_num]['human'] = human_list
        
            #GREEDY
            greedy_obj = greedy(model, tokenizer)
            greedy_word = greedy_obj.get_greedy_word(context)
            dataset[text_id][word_num]['greedy']['pred'] = greedy_word
            dataset[text_id][word_num]['greedy']['cond_log_prob'] = cond_log_probs_obj.conditional_log_prob_of_word_given_context(greedy_word,context)
        
            #ANCESTRAL
            ancestral_obj = ancestral(model, tokenizer)
            list_samples = ancestral_obj.create_sampled_model_distribution(context, log_prob_context)
            dataset[text_id][word_num]['ancestral_samples'] = list_samples
        
            #NUCLEUS
            top_p = [0.7, 0.8, 0.9]
            for p in top_p:
                nucleus_obj = nucleus(p, model, tokenizer)
                list_nucleus = nucleus_obj.create_nucleus_model_distribution(context, log_prob_context)
                dataset[text_id][word_num]['nucleus_samples']['top-p'][p] = list_nucleus
    
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
    file_name = f"Paragraphs-{lower_bound}-{upper_bound - 1}.json"

    with open(file_name, 'w') as fp:
       json.dump(dict_dataset, fp)

input_data = os.path.join(os.getcwd(), 'Provo_Corpus.tsv') 
initiate_dataset_construction(input_data, pre_trained_model = "gpt2", lower_bound = 45, upper_bound = 48)
