import numpy as np
import torch
import torch.utils.data as data

from collections import Counter
import random
import os
import json

def get_provo_data():
    """A function that takes all .json files we created with info for the Provo Corpus
    and merges it into one dictionary"""
    
    #We merge all information in one dictionary
    # Each data point corresponds to all the information relevant to us for a given context in Provo Corpus
    joint_dict = {}

    #input_data = os.listdir(os.path.join(os.getcwd(), 'output') )
    input_data = ['Paragraphs-1-1.json', 'Paragraphs-2-2.json', 'Paragraphs-3-3.json', 'Paragraphs-4-4.json', 
    'Paragraphs-5-9.json', 'Paragraphs-10-14.json', 'Paragraphs-15-19.json', 'Paragraphs-20-24.json', 'Paragraphs-25-29.json',
    'Paragraphs-30-34.json', 'Paragraphs-35-39.json', 'Paragraphs-40-44.json', 'Paragraphs-45-47.json', 'Paragraphs-48-50.json',
    'Paragraphs-51-53.json', 'Paragraphs-54-55.json']
    #os.chdir(os.path.join(os.getcwd(), 'output'))
    os.chdir(os.getcwd())

    count = 0
    for filename in input_data:
        f = open(filename)
        data = json.load(f)
        f.close()

        for text_id in data.keys():
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

class oracle:
    def __init__(self):
        self.seed1 = 0
        self.seed2 = 1

    def sample_oracle_without_replacement_disjoint_groups(self, data_per_context):
        #Create a list with all human answers in a flattened out list ['are', 'are', 'they', ..., 'one']
        list_words_reps = [ [x['pred']]*int(x['count']) for x in data_per_context['human']]
        words_list = [item for sublist in list_words_reps for item in sublist]

        random.seed(self.seed1)
        subset1 = random.sample(words_list, len(words_list)//2)
        subset2 = words_list.copy()
        for item in subset1:
            subset2.remove(item)
        
        return subset1, subset2

    def sample_oracle_with_replacement_bootstrapping(self, data_per_context):
        #Create a list with all human answers in a flattened out list ['are', 'are', 'they', ..., 'one']
        list_words_reps = [ [x['pred']]*int(x['count']) for x in data_per_context['human']]
        words_list = [item for sublist in list_words_reps for item in sublist]

        random.seed(self.seed1)
        probs = [1]* len(words_list) #all words have the same probability to be chosen
        subset1 = random.choices(population = words_list, weights = probs, k = len(words_list)//2)

        random.seed(self.seed2)
        subset2 = random.choices(words_list, weights = probs, k = len(words_list)//2)

        return subset1, subset2
    
    def create_oracle_dist(self, data_per_context, oracle):
        """Receives a list of words that belong to the oracle distribution (subset of human votes) and creates a distribution
        by retrieving all human words for the given instance and allocating the respective probability from the counts"""
        human_words = [ x['pred'] for x in data_per_context['human']]

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


class ECE:
    """On an instance level, we obtain the data necessary to compute ECE, where we consider as a true label
    (for computing accuracy) both the human majority word and the original word"""
    def __init__(self, data_per_context):
        self.data_per_context = data_per_context

    def get_ece_data(self, words, probs):
        """Considering the estimator distribution, we obtain the word (and its confidence) with the maximum 
        probability. For computing accuracy we consider if this word matches the true label (which we consider for 
        both the cases where they are either the original text word and human majority word"""
        #what to do in cases of same votes?
        human_major_word = max(self.data_per_context['human'], key=lambda x:x['count'])['pred']
        original_text_word = self.data_per_context['original']['pred']
                
        p_max_word = words[torch.argmax(probs).item()]
    
        human_maj = (torch.max(probs).item(), int(p_max_word == human_major_word)) 
        orig_text = (torch.max(probs).item(), int(p_max_word == original_text_word)) 
    
        return human_maj, orig_text

class TVD:
    """On a word level, we compute TVD"""
    
    def __init__(self, data_per_context):
        self.data_per_context = data_per_context

    def get_tvd_per_instance(self, words, oracle_probs):
        """Given the distribution (from the model), we retrieve the human distribution for the same words,
        and then compute TVD for the instance level"""
        human_probs = []
        total_human_counts = sum([x['count'] for x in self.data_per_context['human']])
        
        #For the biased distribution we know that the model distribution includes all human words (by design), hence
        #for all of words in the model distribution, we can create the human distribution by either retrieving 
        # the human probability (count/total counts), or setting it is zero
        for word in words:
            human_count = next((x['count'] for x in self.data_per_context['human'] if x['pred'] == word), 0)
            human_prob = human_count/total_human_counts
            human_probs.append(human_prob)

        tvd = np.sum(np.abs(np.array(human_probs) - np.array(oracle_probs)))/2
        return(tvd)

class Entropy:
    def __init__(self, data_per_context):
        self.data_per_context = data_per_context

    def get_entropy_diff_per_instance(self, words, oracle_probs):
        """Given the distribution (from the model), we retrieve the human distribution for the same words,
        and then compute entropy of the humans, the model and their difference at the instance level"""
        human_probs = []
        total_human_counts = sum([x['count'] for x in self.data_per_context['human']])
        
        for word in words:
            human_count = next((x['count'] for x in self.data_per_context['human'] if x['pred'] == word), 0)
            human_prob = human_count/total_human_counts
            human_probs.append(human_prob)
        
        #For zero probability values of p in p log p/q, the contribution to the KL div is thought to be 0, hence we take only
        #non zero p values into account and the respective q values
        human_dist_array = np.array(human_probs)
        oracle_dist_array = np.array(oracle_probs)
        non_zero_human_dist = human_dist_array[human_dist_array > 0]
        non_zero_oracle_dist = oracle_dist_array[oracle_dist_array > 0]

        entropy_human_dist = np.sum(np.multiply(non_zero_human_dist, np.log(non_zero_human_dist)))
        entropy_model_dist = np.sum(np.multiply(non_zero_oracle_dist, np.log(non_zero_oracle_dist)))
        diff_ent = entropy_model_dist - entropy_human_dist
        return(diff_ent)


def get_metrics_data(dataset):
    """Function that iterates over all data points and obtains data relevant to 
    computing calibration metrics (ECE, TVD, KL-Divergence and Entropy difference)"""
        
    dict_out = {}
    dict_out['oracle1_repl_dict'] = {'ece_val_human_maj':[], 'ece_val_orig_text':[], 'tvd_val':[], 'ent_diff_val': []}
    dict_out['oracle2_repl_dict'] = {'ece_val_human_maj':[], 'ece_val_orig_text':[], 'tvd_val':[], 'ent_diff_val': []}
    dict_out['oracle1_no_repl_dict'] = {'ece_val_human_maj':[], 'ece_val_orig_text':[], 'tvd_val':[], 'ent_diff_val': []}
    dict_out['oracle2_no_repl_dict'] = {'ece_val_human_maj':[], 'ece_val_orig_text':[], 'tvd_val':[], 'ent_diff_val': []}

    for key in dataset.keys():
        data_per_context = dataset[key]

        sample_oracle_obj = oracle()
        ece_obj = ECE(data_per_context)
        tvd_obj = TVD(data_per_context)
        ent_obj = Entropy(data_per_context)
            
        oracle1_repl, oracle2_repl = sample_oracle_obj.sample_oracle_without_replacement_disjoint_groups(data_per_context)
        oracle1_no_repl, oracle2_no_repl = sample_oracle_obj.sample_oracle_with_replacement_bootstrapping(data_per_context)
        #Obtaining distributions for all oracle estimators (list of words and a list of their respective probs)

        l = ['oracle1_repl', 'oracle2_repl', 'oracle1_no_repl', 'oracle2_no_repl']
        d = {'oracle1_repl': oracle1_repl,
             'oracle2_repl': oracle2_repl,
             'oracle1_no_repl': oracle1_no_repl,
             'oracle2_no_repl': oracle2_no_repl}

        for item in l:
            var_name = item + '_dict'

            words, probs = sample_oracle_obj.create_oracle_dist(data_per_context, d[item])

            #ECE results
            ece_val_human_maj, ece_val_orig_text = ece_obj.get_ece_data(words, probs)
            dict_out[var_name]['ece_val_human_maj'] += [ece_val_human_maj]
            dict_out[var_name]['ece_val_orig_text'] += [ece_val_orig_text]

            #TVD results
            tvd_val = tvd_obj.get_tvd_per_instance(words, probs)
            dict_out[var_name]['tvd_val'] += [tvd_val]
                
            #Entropy difference results
            ent_diff_val = ent_obj.get_entropy_diff_per_instance(words, probs)
            dict_out[var_name]['ent_diff_val'] += [ent_diff_val]

    return dict_out

def calculate_ECE(conf_acc, ece_bins = 10):
    """Function that given a list of tuples including the confidence of each prediction and if it matches the
    true label computes the ECE. To do that, we split the confidence space (0,1) in bins, separate predictions
    according to the bins, calculate the average confidence per bin and the accuracy per bin and take their weighted
    average."""
    bins = ece_bins
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
    
def calculate_list_metrics(list_met):
    """For a list of instance level metric computations, we compute their key metrics - mean, standard deviation,
    min and max values"""
    array_met = np.array(list_met)
    avg_met = np.mean(array_met)
    std_met = np.std(array_met)
    min_met = np.min(array_met)
    max_met = np.max(array_met)
    return (avg_met, std_met, min_met, max_met)

def calculate_metrics(dataset):
    dict_results = {}
    dict_out = get_metrics_data(dataset)
    
    for key in dict_out.keys():
        dict_results[key] = {}

        #ECE
        dict_results[key]['ECE_dist_human_maj_truth'] = calculate_ECE(dict_out[key]['ece_val_human_maj'])
        dict_results[key]['ECE_dist_orig_word_truth'] = calculate_ECE(dict_out[key]['ece_val_orig_text'])
        
        #TVD
        dict_results[key]['TVD_per_instance'] = dict_out[key]['tvd_val']
        dict_results[key]['TVD_mean_std_min_max'] = calculate_list_metrics(dict_out[key]['tvd_val'])
        
        #Entropy difference
        dict_results[key]['Entropy_difference_per_instance'] = dict_out[key]['ent_diff_val']
        dict_results[key]['Entropy_difference_mean_std_min_max'] = calculate_list_metrics(dict_out[key]['ent_diff_val'])

    return dict_results

dataset = get_provo_data()
print("Size of dataset:", len(dataset))

dict_out = calculate_metrics(dataset)

with open("cal_metrics_oracle.json", "w") as outfile:
    json.dump(dict_out, outfile)