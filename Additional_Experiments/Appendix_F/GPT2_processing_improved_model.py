import numpy as np
import torch
import torch.utils.data as data

import random
import os
import json
from collections import OrderedDict
from collections import Counter

def get_provo_data(input_data):
    """A function that takes all .json files we created with info for the Provo Corpus
    and merges it into one dictionary"""
    
    #We merge all information in one dictionary
    # Each data point corresponds to all the information relevant to us for a given context in Provo Corpus
    joint_dict = {}
    
    count = 0
    for filename in input_data:
        f = open(filename)
        data = json.load(f)
        f.close()

        for text_id in data.keys():
            if (int(text_id) > 0) & (int(text_id) <= 55):
                for word_num in data[text_id].keys():
                    joint_dict[count] = data[text_id][word_num]
                    joint_dict[count]['original_positioning'] = {'text_id':text_id, 'word_num':word_num}
                
                    count = count + 1

    return joint_dict


class Provo_Dataset(data.Dataset):
    
    def __init__(self):
        super().__init__()
        self.joint_dict = get_provo_data()
        
    def __len__(self):
        # Number of data point we have. 
        return len(self.joint_dict.keys())
    
    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        x = self.joint_dict[idx]
        return x

class estimator:
    def __init__(self, data_per_context):
        self.data_per_context = data_per_context
        
    def get_estimator_unbiased(self):
        """For each data point we compute the estimator where the words belong to the unbiased distribution"""
        #Check for failed to generate full-word samples and remove those
        fail = [d for d in self.data_per_context['ancestral_samples'] if d['pred'] == 'Failed to generate word']
        if len(fail) > 0:
            self.data_per_context['ancestral_samples'].remove(fail[0])
        
        #remove duplicates (from lowercasing)
        support_model = [x['pred'].lower() for x in self.data_per_context['ancestral_samples']]
        counts_model = [x['count'] for x in self.data_per_context['ancestral_samples']]
        dist = [[support_model[i], counts_model[i]] for i in range(len(counts_model))]
        words = set(map(lambda x:x[0], dist))
        merged_list = [[y[1] for y in dist if y[0]==x] for x in words]
        merged_counts = [sum(x) for x in merged_list]
        probs = torch.Tensor([x/sum(merged_counts) for x in merged_counts])

        return list(words), probs

    def get_estimator_biased(self):
        """For each data point we compute the biased estimator"""

        #putting together all elements of the biased estimator
        list_words = []
        list_words.append(self.data_per_context['original'])
        list_words.append(self.data_per_context['greedy'])
        list_words = list_words + self.data_per_context['human']
        list_words = list_words + self.data_per_context['ancestral_samples']
        for p in self.data_per_context['nucleus_samples']['top-p'].keys():
            list_words = list_words + self.data_per_context['nucleus_samples']['top-p'][p]
        
        #remove duplicate entries for the same prediction
        list_unique_words = OrderedDict([(d['pred'], d['cond_log_prob']) for d in list_words]).items()
        
        #All words and scores that belong to the biased estimator distribution
        words = [ x[0] for x in list_unique_words ] 
        scores = torch.Tensor([ x[1] for x in list_unique_words ])
        log_probs = scores - torch.logsumexp(scores, -1, keepdims=True)  #stable softmax
        probs = torch.exp(log_probs) #log probs to probs

        return words, probs


class oracle:
    def __init__(self, data_per_context):
        self.data_per_context = data_per_context

        self.seed = 0

    def sample_oracle_without_replacement_disjoint_groups(self, N = 20):
        """We create two disjoint subsets of the human distribution by sampling without replacement from
        the human distribution"""
        #Create a list with all human answers in a flattened out list ['are', 'are', 'they', ..., 'one']
        list_words_reps = [ [x['pred']]*int(x['count']) for x in self.data_per_context['human']]
        words_list = [item for sublist in list_words_reps for item in sublist]

        random.seed(self.seed)

        #if the length of the list is odd, we remove one element at random to make the list even,
        #since we want the two disjoint subsets to be of equal length
        if (len(words_list) % 2 == 1): 
            remove_word = random.sample(words_list, 1)
            words_list.remove(remove_word[0])

        #We sample the words that will belong in the first subset and create the second subset by removing
        #from the full word list the ones sampled in the first subset
        subset1 = random.sample(words_list, len(words_list)//2)
        subset2 = words_list.copy()
        for item in subset1:
            subset2.remove(item)
        
        return subset1, subset2
    
    def sample_oracle_with_replacement_bootstrapping(self, N = 20):
        """We create two bootstrapped sets (not necessarily disjoint) of oracle classifiers by sampling 
        N samples with replacement from the full human distribution"""
        #Create a list with all human answers in a flattened out list ['are', 'are', 'they', ..., 'one']
        list_words_reps = [ [x['pred']]*int(x['count']) for x in self.data_per_context['human']]
        words_list = [item for sublist in list_words_reps for item in sublist]

        random.seed(self.seed)
        probs = [1]* len(words_list) #all words have the same probability to be chosen
        subset1 = random.choices(population = words_list, weights = probs, k = N)

        random.seed(self.seed + 1)
        subset2 = random.choices(words_list, weights = probs, k = N)

        return subset1, subset2
    
    def create_oracle_dist(self, oracle):
        """Receives a list of words that belong to the oracle distribution (subset of human votes) and creates a distribution
        by retrieving all human words for the given instance and allocating the respective probability from the counts"""
        human_words = [ x['pred'] for x in self.data_per_context['human']]

        dict_word_counts = Counter(oracle)
        dist_oracle = []

        for word in human_words:
            try:
                count = dict_word_counts[word]
            except:
                count = 0 
            prob = count/len(oracle)
            dist_oracle.append(prob)
        
        return(human_words, torch.Tensor(dist_oracle))

class TVD:
    """On a word level, we compute TVD"""
    
    def __init__(self, data_per_context):
        self.data_per_context = data_per_context

    def compute_TVD(self, probs1, probs2):
        tvd = torch.sum(torch.abs(probs1 - probs2))/2
        return tvd.item()

    def get_tvd_per_instance_for_biased_est_dist_and_oracle(self, model_words, model_probs, oracle_words, oracle_probs):
        """Given the distribution (from the model), we retrieve the human distribution for the same words,
        and then compute TVD for the instance level"""
        
        #We know that the items of the model distribution and the oracle distribution are not currently aligned.
        #Thus, before computing the TVD between them we first need to align the sample space and probabilities between
        #the two distributions
        human_probs = []
        
        #For the biased distribution we know that the model distribution includes all human words (by design), hence
        #for all of words in the model distribution, we can create the human distribution by either retrieving 
        # the human probability (count/total counts), or setting it is zero
        for word in model_words:
            try:
                index_word = oracle_words.index(word)
                human_probs.append(oracle_probs[index_word].item())
            except:
                human_probs.append(0)

        tvd = self.compute_TVD(torch.Tensor(human_probs), torch.Tensor(model_probs))
        return(tvd)

    def get_tvd_per_instance_for_unbiased_est_dist_and_oracle(self, model_words, model_probs, oracle_words, oracle_probs):
        """Given the distribution (from the model), we retrieve the human distribution for the same words,
        and then compute TVD for the instance level"""

        #We know that the items of the model distribution and the oracle distribution are not currently aligned
        #Thus, before computing the TVD between them we first need to align the sample space and probabilities between
        #the two distributions
        human_probs = []
        
        list_model_probs = model_probs.tolist()
        list_words = model_words.copy()

        #For the unbiased distributions, the sampled words may not necessarily include all human words. Hence,
        # before creating the human distribution, we add to the model one the ones that are missing with a respective 
        # probability of zero
        list_missing = list(set([x['pred'] for x in self.data_per_context['human']]) - set(list_words)) #set of human words that are not in the model distribution words

        for missing_word in list_missing:
            list_words.append(missing_word)
            list_model_probs.append(0)
        
        #Similarly to the biased dist., we iterate over all words and the human dist. probabilities are either the retrieved
        #probability from the oracle dist. or zero
        for word in list_words:
            try:
                index_word = oracle_words.index(word)
                human_probs.append(oracle_probs[index_word].item())
            except:
                human_probs.append(0)

        tvd = self.compute_TVD(torch.Tensor(human_probs), torch.Tensor(list_model_probs))
        return(tvd)

class metrics:
    def __init__(self, dataset):
        self.dataset = dataset

        self.ece_bins = 10
        self.n_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.seed = 0

    def get_metrics_data(self):
        """Function that iterates over all data points and obtains data relevant to 
        computing calibration metrics (ECE, TVD, KL-Divergence and Entropy difference)"""
        
        dict_out = {'tvd_val': {'tvd_per_instance_biased_model_human1' : [],
                                'tvd_per_instance_unbiased_model_human1' : [],
                                'tvd_per_instance_human2_human1' : []}} 

        for key in self.dataset.keys():
            data_per_context = self.dataset[key]

            oracle_obj = oracle(data_per_context)
            est = estimator(data_per_context)
            tvd_obj = TVD(data_per_context)
            
            #Obtaining distributions for all oracle estimators (list of words and a list of their respective probs)
            human_1, human_2 = oracle_obj.sample_oracle_without_replacement_disjoint_groups()
            
            human1_words, human1_probs = oracle_obj.create_oracle_dist(human_1)
            human2_words, human2_probs = oracle_obj.create_oracle_dist(human_2)
            
            #Obtaining distributions for biased and unbiased estimators (list of words and a list of their respective probs)
            model_unbiased_words, model_unbiased_probs = est.get_estimator_unbiased()
            model_biased_words, model_biased_probs = est.get_estimator_biased()

            #Total Variation Distance Data
            tvd_human2_human1 = tvd_obj.compute_TVD(human2_probs, human1_probs)

            tvd_per_instance_biased_model_human1 = tvd_obj.get_tvd_per_instance_for_biased_est_dist_and_oracle(model_biased_words, model_biased_probs, human1_words, human1_probs)
            tvd_per_instance_unbiased_model_human1 = tvd_obj.get_tvd_per_instance_for_unbiased_est_dist_and_oracle(model_unbiased_words, model_unbiased_probs, human1_words, human1_probs)
            
            dict_out['tvd_val']['tvd_per_instance_biased_model_human1'].append(tvd_per_instance_biased_model_human1)
            dict_out['tvd_val']['tvd_per_instance_unbiased_model_human1'].append(tvd_per_instance_unbiased_model_human1)
            dict_out['tvd_val']['tvd_per_instance_human2_human1'].append(tvd_human2_human1)
        
        return dict_out
    
    def calculate_metrics(self):
        dict_results = {}
        dict_out = self.get_metrics_data()
        
        for n in self.n_list:
            dict_results[n] = {}
            #Setting a random seed for each value of n 
            np.random.seed(int(n*10))
            
            #TVD
            tvd_oracle_eval = dict_out['tvd_val']['tvd_per_instance_human2_human1']

            tvd_model_eval_biased = dict_out['tvd_val']['tvd_per_instance_biased_model_human1']
            
            tvd_mixed_eval_biased = np.where(np.random.random((len(tvd_oracle_eval))) > n, tvd_model_eval_biased, tvd_oracle_eval)
            dict_results[n]['tvd_biased'] = list(tvd_mixed_eval_biased)
            
            tvd_model_eval_unbiased = dict_out['tvd_val']['tvd_per_instance_unbiased_model_human1']
            
            tvd_mixed_eval_unbiased = np.where(np.random.random((len(tvd_oracle_eval))) > n, tvd_model_eval_unbiased, tvd_oracle_eval)
            dict_results[n]['tvd_unbiased'] = list(tvd_mixed_eval_unbiased)

        return dict_results

os.chdir(os.path.join(os.getcwd(), '/TVD/GPT2_FT/Generations'))
# os.chdir(os.path.join(os.getcwd(), '/TVD/GPT2/Generations'))

input_data = ['Paragraphs-31-33_fine_tuned.json', 'Paragraphs-34-36_fine_tuned.json', 
              'Paragraphs-37-38_fine_tuned.json', 'Paragraphs-39-40_fine_tuned.json']


d = get_provo_data(input_data)

metrics_obj = metrics(dataset = d)        

dict_results = metrics_obj.calculate_metrics()

with open("cal_metrics_improved_model.json", "w") as outfile:
    json.dump(dict_results, outfile)