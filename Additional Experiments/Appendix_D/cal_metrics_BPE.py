import numpy as np
import torch
import torch.utils.data as data

import random
import os
import json
from collections import OrderedDict
from collections import Counter

def get_provo_data():
    """A function that takes all .json files we created with info for the Provo Corpus
    and merges it into one dictionary"""
    
    #We merge all information in one dictionary
    # Each data point corresponds to all the information relevant to us for a given context in Provo Corpus
    joint_dict = {}

    os.chdir(os.path.join(os.getcwd(), 'output/data_preprocessing_bpes'))
    input_data = ['Paragraphs_BPE-1-1.json',
    'Paragraphs_BPE-2-2.json', 'Paragraphs_BPE-3-3.json', 'Paragraphs_BPE-4-4.json',
    'Paragraphs_BPE-5-9.json', 'Paragraphs_BPE-10-14.json', 'Paragraphs_BPE-15-19.json', 'Paragraphs_BPE-20-24.json',
    'Paragraphs_BPE-25-29.json', 'Paragraphs_BPE-30-34.json', 'Paragraphs_BPE-35-39.json', 'Paragraphs_BPE-40-44.json',
    'Paragraphs_BPE-45-47.json', 'Paragraphs_BPE-48-50.json', 'Paragraphs_BPE-51-53.json', 'Paragraphs_BPE-54-55.json']
    #os.chdir(os.path.join(os.getcwd(), 'output'))

    count = 0
    for filename in input_data:
        print(filename)
        f = open(filename, 'r')
        data = json.load(f)
        f.close()
        
        vocab_dict = data['vocab_dict']
        list_keys = list(data.keys())
        list_keys.remove('vocab_dict')
        for text_id in list_keys:
            for word_num in data[text_id].keys():
                joint_dict[count] = data[text_id][word_num]
                joint_dict[count]['original_positioning'] = {'text_id':text_id, 'word_num':word_num}
                
                count = count + 1

    return joint_dict, vocab_dict

def stable_normalisation(scores):
    log_p = scores - torch.logsumexp(scores, -1, keepdims=True)  
    probs = torch.exp(log_p)

    return probs


class estimator:
    def __init__(self, data_per_context):
        self.data_per_context = data_per_context
    
    def create_estimator(self, indeces, counts, support_len):
        #Turning the counts to probabilities
        probs = counts / torch.sum(counts)

        dist = torch.zeros(support_len)
        dist[indeces] = probs

        return dist

    def get_estimator_human(self):
        """For each data point we compute the human distribution - the support of the distribution is 
        just the model vocabulary - all bpes. We return the torch tensor of the distribution's probabilities"""

        indeces = torch.LongTensor([ x['bpe_ind'] for x in self.data_per_context['human'] ])
        scores = torch.Tensor([ x['count'] for x in self.data_per_context['human']])

        dist = self.create_estimator(indeces, scores, len(self.data_per_context['model']['cond_log_prob']))

        return dist

    def get_estimator_oracle(self, oracle):
        """Receives a list of words that belong to the oracle distribution (subset of human votes) and creates a 
        distribution with the relevant probabilities and support the token vocabulary"""
        
        dict_word_counts = Counter(oracle)

        indeces = torch.LongTensor(list(dict_word_counts.keys()))
        scores = torch.Tensor(list(dict_word_counts.values()))

        dist = self.create_estimator(indeces, scores, len(self.data_per_context['model']['cond_log_prob']))
        
        return dist


class oracle:
    def __init__(self, data_per_context, seed):
        self.data_per_context = data_per_context
        self.seed = seed

    def sample_oracle_without_replacement_disjoint_groups(self, N):
        """We create two disjoint subsets  ofthe human distribution by sampling without replacement from
        the human distribution (the two disjoing subsets can be comprised by either 10 or 20 samples"""
        #Create a list with all human answers in a flattened out list ['are', 'are', 'they', ..., 'one']
        list_words_reps = [ [str(x['bpe_ind'])]*int(x['count']) for x in self.data_per_context['human']]
        words_list = [item for sublist in list_words_reps for item in sublist]

        random.seed(self.seed)

        #if the length of the list is odd, we remove one element at random to make the list even,
        #since we want the two disjoint subsets to be of equal length
        if (len(words_list) % 2 == 1): 
            remove_word = random.sample(words_list, 1)
            words_list.remove(remove_word[0])

        #We sample the words that will belong in the first subset and create the second subset by removing
        #from the full word list the ones sampled in the first subset
        subset1 = random.sample(words_list, N)
        if N == 20:
            subset2 = words_list.copy()
            for item in subset1:
                subset2.remove(item)
        elif N == 10:
            subset_left = words_list.copy()
            for item in subset1:
                subset_left.remove(item)
            subset2 = random.sample(subset_left, N)
            
        return [int(x) for x in subset1], [int(x) for x in subset2]
    
    def sample_oracle_with_replacement_bootstrapping(self, N):
        """We create two bootstrapped sets (not necessarily disjoint) of oracle classifiers by sampling 
        N samples with replacement from the full human distribution"""
        #Create a list with all human answers in a flattened out list ['are', 'are', 'they', ..., 'one']
        list_words_reps = [ [str(x['bpe_ind'])]*int(x['count']) for x in self.data_per_context['human']]
        words_list = [item for sublist in list_words_reps for item in sublist]

        random.seed(self.seed)
        probs = [1]* len(words_list) #all words have the same probability to be chosen
        subset1 = random.choices(population = words_list, weights = probs, k = N)

        random.seed(self.seed + 1)
        subset2 = random.choices(words_list, weights = probs, k = N)
        
        return [int(x) for x in subset1], [int(x) for x in subset2]

class ECE:
    """On an instance level, we obtain the data necessary to compute ECE, where we consider as a true label
    (for computing accuracy) both the human majority word and the original word"""
    def __init__(self, data_per_context):
        self.data_per_context = data_per_context

    def get_ece_data(self, probs):
        """Considering the estimator distribution, we obtain the word (and its confidence) with the maximum 
        probability. For computing accuracy we consider if this word matches the true label (which we consider for 
        both the cases where they are either the original text word and human majority word"""

        #what to do in cases of same votes?
        human_major_bpe = max(self.data_per_context['human'], key=lambda x:x['count'])['bpe_ind']

        #original_text_word = self.data_per_context['original']['bpe_ind']

        p_max_ind = torch.argmax(probs).item()

        human_maj = (torch.max(probs).item(), int(p_max_ind == human_major_bpe)) 
        #orig_text = (torch.max(probs).item(), int(p_max_ind == original_text_word)) 

        return human_maj #, orig_text

class TVD:
    """On a word level, we compute TVD"""
    def __init__(self):
        pass

    def compute_TVD(self, probs1, probs2):
        tvd = torch.sum(torch.abs(probs1 - probs2))/2
        return tvd.item()

class KL_div:
    """On a word level, we compute KL-divergence"""
    
    def __init__(self):
        pass

    def compute_kl_div(self, probs1, probs2):
        """Receives two Torch tensors of two probability distributions and computes their KL-Divergence"""
        non_zero_probs1 = probs1[probs1 > 0]
        non_zero_probs2 = probs2[probs1 > 0]

        kl_div = torch.sum(torch.multiply(non_zero_probs1, torch.log(torch.divide(non_zero_probs1, non_zero_probs2))))
        return kl_div.item()

class Entropy:
    def __init__(self):
        pass

    def compute_ent_diff(self, probs1, probs2):
        """Receives two Torch tensors of two probability distributions and computes their absolute 
        difference of entropy"""
        
        #For zero probability values of p in p log p, the contribution to entropy is 0, hence we take only
        #non zero p values into account
        non_zero_probs1 = probs1[probs1 > 0]
        non_zero_probs2 = probs2[probs2 > 0]

        entropy_probs1 = torch.sum(torch.multiply(non_zero_probs1, torch.log(non_zero_probs1)))
        entropy_probs2 = torch.sum(torch.multiply(non_zero_probs2, torch.log(non_zero_probs2)))
        diff_ent = torch.abs(entropy_probs1 - entropy_probs2)

        return diff_ent.item()


class metrics:
    def __init__(self, dataset):
        self.dataset = dataset

        self.ece_bins = 10

    def calculate_ECE(self, conf_acc):
        """Function that given a list of tuples including the confidence of each prediction and if it matches the
        true label computes the ECE. To do that, we split the confidence space (0,1) in bins, separate predictions
        according to the bins, calculate the average confidence per bin and the accuracy per bin and take their weighted
        average."""
        bins = self.ece_bins
        conf_acc_array = np.array(conf_acc)

        N = conf_acc_array.shape[0]

        sum_bin = 0
        for i in np.arange(0, 1, 1/bins):
            #getting all points which belong to the relevant bin - given their 
            bin_contents = conf_acc_array[np.where((conf_acc_array[:,0] >= i) & (conf_acc_array[:,0] < (i + 1/bins)))]
            n_bin = bin_contents[:,0].shape[0]
            if n_bin > 0: #if the bin is non empty
                avg_conf = np.sum(bin_contents[:,0]) / n_bin
                acc = np.sum(bin_contents[:,1]) / n_bin
                sum_bin = sum_bin + abs(avg_conf - acc) * n_bin / N
        
        ece_val = sum_bin
        return(ece_val) 

    def get_metrics_data(self):
        """Function that iterates over all data points and obtains data relevant to 
        computing calibration metrics (ECE, TVD, KL-Divergence and Entropy difference)"""
        
        conf_acc_human_maj_model_human = []
        conf_acc_human_maj_oracle_dis_1_human = []
        conf_acc_human_maj_oracle_dis_2_human = []
        conf_acc_human_maj_oracle_boot_1_human = []
        conf_acc_human_maj_oracle_boot_2_human = []
        
        #conf_acc_orig_text = []
    
        tvd_model_human = []
        tvd_oracle_dis_1_human = []
        tvd_oracle_dis_2_human = []
        tvd_oracle_boot_1_human = []
        tvd_oracle_boot_2_human = []

        kl_div = []

        ent_diff_model_human = []
        ent_diff_oracle_dis_1_human = []
        ent_diff_oracle_dis_2_human = []
        ent_diff_oracle_boot_1_human = []
        ent_diff_oracle_boot_2_human = []

        for key in self.dataset.keys():
            data_per_context = self.dataset[key]
            logits = torch.Tensor(data_per_context['model']['cond_log_prob'])
            model_probs = stable_normalisation(logits)

            est = estimator(data_per_context)
            #Obtaining distributions for biased and unbiased estimators (list of words and a list of their respective probs)
            human_probs = est.get_estimator_human()

            orac = oracle(data_per_context, seed=0)
            subset1, subset2 = orac.sample_oracle_without_replacement_disjoint_groups(20)

            oracle_dis_probs_1 = est.get_estimator_oracle(subset1)
            oracle_dis_probs_2 = est.get_estimator_oracle(subset2)

            subset1, subset2 = orac.sample_oracle_with_replacement_bootstrapping(20)

            oracle_boot_probs_1 = est.get_estimator_oracle(subset1)
            oracle_boot_probs_2 = est.get_estimator_oracle(subset2)

            #Expected Calibration Error Data
            ece = ECE(data_per_context)

            #Model - to full human dist
            conf_acc_human_maj_model_human.append( ece.get_ece_data(model_probs) )
            conf_acc_human_maj_oracle_dis_1_human.append( ece.get_ece_data(oracle_dis_probs_1) )
            conf_acc_human_maj_oracle_dis_2_human.append( ece.get_ece_data(oracle_dis_probs_2) )
            conf_acc_human_maj_oracle_boot_1_human.append( ece.get_ece_data(oracle_boot_probs_1) )
            conf_acc_human_maj_oracle_boot_2_human.append( ece.get_ece_data(oracle_boot_probs_2) )
            
            #human_maj, orig_text = ece.get_ece_data(model_probs)
            #conf_acc_orig_text.append( orig_text )
            
            #Total Variation Distance Data
            tvd_obj = TVD()
            tvd_model_human_per_instance = tvd_obj.compute_TVD(human_probs, model_probs)
            tvd_oracle_dis_1_human_per_instance = tvd_obj.compute_TVD(human_probs, oracle_dis_probs_1)
            tvd_oracle_dis_2_human_per_instance = tvd_obj.compute_TVD(human_probs, oracle_dis_probs_2)
            tvd_oracle_boot_1_human_per_instance = tvd_obj.compute_TVD(human_probs, oracle_boot_probs_1)
            tvd_oracle_boot_2_human_per_instance = tvd_obj.compute_TVD(human_probs, oracle_boot_probs_2)
            
            tvd_model_human.append(tvd_model_human_per_instance)
            tvd_oracle_dis_1_human.append(tvd_oracle_dis_1_human_per_instance)
            tvd_oracle_dis_2_human.append(tvd_oracle_dis_2_human_per_instance)
            tvd_oracle_boot_1_human.append(tvd_oracle_boot_1_human_per_instance)
            tvd_oracle_boot_2_human.append(tvd_oracle_boot_2_human_per_instance)

            #KL-Divergence
            kl_div_obj = KL_div()
            kl_div_per_instance = kl_div_obj.compute_kl_div(human_probs, model_probs)
            kl_div.append(kl_div_per_instance)
            
            #Entropy difference
            ent_diff_obj = Entropy()
            ent_diff_model_human_per_instance = ent_diff_obj.compute_ent_diff(human_probs, model_probs)
            ent_diff_oracle_dis_1_human_per_instance = ent_diff_obj.compute_ent_diff(human_probs, oracle_dis_probs_1)
            ent_diff_oracle_dis_2_human_per_instance = ent_diff_obj.compute_ent_diff(human_probs, oracle_dis_probs_2)
            ent_diff_oracle_boot_1_human_per_instance = ent_diff_obj.compute_ent_diff(human_probs, oracle_boot_probs_1)
            ent_diff_oracle_boot_2_human_per_instance = ent_diff_obj.compute_ent_diff(human_probs, oracle_boot_probs_2)
            
            ent_diff_model_human.append(ent_diff_model_human_per_instance)
            ent_diff_oracle_dis_1_human.append(ent_diff_oracle_dis_1_human_per_instance)
            ent_diff_oracle_dis_2_human.append(ent_diff_oracle_dis_2_human_per_instance)
            ent_diff_oracle_boot_1_human.append(ent_diff_oracle_boot_1_human_per_instance)
            ent_diff_oracle_boot_2_human.append(ent_diff_oracle_boot_2_human_per_instance)


        dict_results = {'ECE_human_maj_truth': {'model_human':self.calculate_ECE(conf_acc_human_maj_model_human),
                                                'oracle_dis_1_human':self.calculate_ECE(conf_acc_human_maj_oracle_dis_1_human),
                                                'oracle_dis_2_human':self.calculate_ECE(conf_acc_human_maj_oracle_dis_2_human),
                                                'oracle_boot_1_human':self.calculate_ECE(conf_acc_human_maj_oracle_boot_1_human),
                                                'oracle_boot_2_human':self.calculate_ECE(conf_acc_human_maj_oracle_boot_2_human)},
                        'TVD_per_instance': {'tvd_model_human':tvd_model_human,
                                            'tvd_oracle_dis_1_human': tvd_oracle_dis_1_human,
                                            'tvd_oracle_dis_2_human': tvd_oracle_dis_2_human,
                                            'tvd_oracle_boot_1_human': tvd_oracle_boot_1_human,
                                            'tvd_oracle_boot_2_human': tvd_oracle_boot_2_human},
                        'KL_Divergenece_per_instance': kl_div,
                        'Entropy_difference_per_instance': {'ent_diff_model_human':ent_diff_model_human,
                                                            'ent_diff_oracle_dis_1_human':ent_diff_oracle_dis_1_human,
                                                            'ent_diff_oracle_dis_2_human':ent_diff_oracle_dis_2_human,
                                                            'ent_diff_oracle_boot_1_human':ent_diff_oracle_boot_1_human,
                                                            'ent_diff_oracle_boot_2_human':ent_diff_oracle_boot_2_human}}

        return dict_results


joint_dict, vocab_dict = get_provo_data()
print("Size of dataset:", len(joint_dict))

metrics_obj = metrics(joint_dict)        
dict_results = metrics_obj.get_metrics_data()

with open("cal_metrics_bpe.json", "w") as outfile:
   json.dump(dict_results, outfile)