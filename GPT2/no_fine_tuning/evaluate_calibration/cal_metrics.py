import numpy as np
import torch
import torch.utils.data as data

import os
import json
from collections import OrderedDict

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
    os.chdir(os.path.join(os.getcwd(), 'output'))

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


class estimator:
    def __init__(self, data_per_context):
        self.data_per_context = data_per_context
        
    def get_estimator_unbiased(self):
        """For each data point we compute the unbiased estimator"""
        #lists with all conditional words and counts from the ancestral samples
        words = [ x['pred'] for x in self.data_per_context['ancestral_samples'] ]
        scores = torch.Tensor([ x['count'] for x in self.data_per_context['ancestral_samples']])
        #Turning the counts to probabilities
        probs = scores / torch.sum(scores)
    
        return words, probs

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

    def get_tvd_per_instance_for_biased_est_dist(self, words, model_probs):
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

        tvd = np.sum(np.abs(np.array(human_probs) - np.array(model_probs)))/2
        return(tvd)

    def get_tvd_per_instance_for_unbiased_est_dist(self, words, model_probs):
        """Given the distribution (from the model), we retrieve the human distribution for the same words,
        and then compute TVD for the instance level"""
        human_probs = []
        total_human_counts = sum([x['count'] for x in self.data_per_context['human']])
        
        list_model_probs = model_probs.tolist()
        list_words = words.copy()

        #For the unbiased distributions, the sampled words may not necessarily include all human words. Hence,
        # before creating the human distribution, we add to the model one the ones that are missing with a respective 
        # probability of zero
        list_missing = list(set([x['pred'] for x in self.data_per_context['human']]) - set(list_words)) #set of human words that are not in the model distribution words
        
        for missing_word in list_missing:
            list_words.append(missing_word)
            list_model_probs.append(0)
        
        for word in list_words:
            human_count = next((x['count'] for x in self.data_per_context['human'] if x['pred'] == word), 0)
            human_prob = human_count/total_human_counts
            human_probs.append(human_prob)

        tvd = np.sum(np.abs(np.array(human_probs) - np.array(list_model_probs)))/2
        return(tvd)


class KL_div:
    """On a word level, we compute KL-divergence"""
    
    def __init__(self, data_per_context):
        self.data_per_context = data_per_context

    def get_kl_div_biased_per_instance(self, words, model_probs):
        """Given the distribution (from the model), we retrieve the human distribution for the same words,
        and then compute TVD for the instance level"""
        human_probs = []
        total_human_counts = sum([x['count'] for x in self.data_per_context['human']])
        
        for word in words:
            human_count = next((x['count'] for x in self.data_per_context['human'] if x['pred'] == word), 0)
            human_prob = human_count/total_human_counts
            human_probs.append(human_prob)
        
        #For zero probability values of p in p log p/q, the contribution to the KL div is thought to be 0, hence we take only
        #non zero p values into account and the respective q values
        human_dist_array = np.array(human_probs)
        model_dist_array = np.array(model_probs)
        non_zero_human_dist = human_dist_array[human_dist_array > 0]
        non_zero_model_dist = model_dist_array[human_dist_array > 0]

        kl_div = np.sum(np.multiply(non_zero_human_dist, np.log(np.divide(non_zero_human_dist, non_zero_model_dist))))
        return(kl_div)

    def get_kl_div_unbiased_per_instance(self, words, model_probs):
        """Given the distribution (from the model), we retrieve the human distribution for the same words,
        and then compute TVD for the instance level"""
        human_probs = []
        total_human_counts = sum([x['count'] for x in self.data_per_context['human']])
        
        list_words = words.copy()
        list_model_probs = model_probs.tolist()

        #For the unbiased distributions, the sampled words may not necessarily include all human words. Hence,
        # before creating the human distribution, we add to the model one the ones that are missing with a respective 
        # probability of zero
        list_missing = list(set([x['pred'] for x in self.data_per_context['human']]) - set(list_words)) #set of human words that are not in the model distribution words
        
        for missing_word in list_missing:
            list_words.append(missing_word)
            list_model_probs.append(0)

        for word in list_words:
            human_count = next((x['count'] for x in self.data_per_context['human'] if x['pred'] == word), 0)
            human_prob = human_count/total_human_counts
            human_probs.append(human_prob)
                
        #For zero probability values of p in p log p/q, the contribution to the KL div is thought to be 0, hence we take only
        #non zero p values into account and the respective q values
        human_dist_array = np.array(human_probs)
        model_dist_array = np.array(list_model_probs)
        non_zero_human_dist = human_dist_array[human_dist_array > 0] #we are only interested in the entries with non-zero p (aka human dist)
        non_zero_model_dist = model_dist_array[human_dist_array > 0]

        kl_div = np.sum(np.multiply(non_zero_human_dist, np.log(np.divide(non_zero_human_dist, non_zero_model_dist))))
        return(kl_div)

class Entropy:
    def __init__(self, data_per_context):
        self.data_per_context = data_per_context

    def get_entropy_biased_per_instance(self, words, model_probs):
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
        model_dist_array = np.array(model_probs)
        non_zero_human_dist = human_dist_array[human_dist_array > 0]
        non_zero_model_dist = model_dist_array[model_dist_array > 0]

        entropy_human_dist = np.sum(np.multiply(non_zero_human_dist, np.log(non_zero_human_dist)))
        entropy_model_dist = np.sum(np.multiply(non_zero_model_dist, np.log(non_zero_model_dist)))
        diff_ent = entropy_model_dist - entropy_human_dist
        return(diff_ent)

    def get_entropy_unbiased_per_instance(self, words, model_probs):
        """Given the distribution (from the model), we retrieve the human distribution for the same words,
        and then compute entropy of the humans, the model and their difference at the instance level"""
        human_probs = []
        total_human_counts = sum([x['count'] for x in self.data_per_context['human']])
        
        list_words = words.copy()
        list_model_probs = model_probs.tolist()

        #For the unbiased distributions, the sampled words may not necessarily include all human words. Hence,
        # before creating the human distribution, we add to the model one the ones that are missing with a respective 
        # probability of zero
        list_missing = list(set([x['pred'] for x in self.data_per_context['human']]) - set(list_words)) #set of human words that are not in the model distribution words
        

        for missing_word in list_missing:
            list_words.append(missing_word)
            list_model_probs.append(0)
        
        for word in list_words:
            human_count = next((x['count'] for x in self.data_per_context['human'] if x['pred'] == word), 0)
            human_prob = human_count/total_human_counts
            human_probs.append(human_prob)
        
        #For zero probability values of p in p log p/q, the contribution to the KL div is thought to be 0, hence we take only
        #non zero p values into account and the respective q values
        human_dist_array = np.array(human_probs)
        model_dist_array = np.array(list_model_probs)
        non_zero_human_dist = human_dist_array[human_dist_array > 0]
        non_zero_model_dist = model_dist_array[model_dist_array > 0]

        entropy_human_dist = np.sum(np.multiply(non_zero_human_dist, np.log(non_zero_human_dist)))
        entropy_model_dist = np.sum(np.multiply(non_zero_model_dist, np.log(non_zero_model_dist)))
        diff_ent = entropy_model_dist - entropy_human_dist
        return(diff_ent)

class metrics:
    def __init__(self, dataset):
        self.dataset = dataset

        self.ece_bins = 10

    def get_metrics_data(self):
        """Function that iterates over all data points and obtains data relevant to 
        computing calibration metrics (ECE, TVD, KL-Divergence and Entropy difference)"""
        
        conf_acc_human_maj_biased = []
        conf_acc_orig_text_biased = []
    
        conf_acc_human_maj_unbiased = []
        conf_acc_orig_text_unbiased = []

        tvd_biased = []
        tvd_unbiased = []

        kl_div_biased = []
        #kl_div_unbiased = []

        entropy_biased = []
        entropy_unbiased = []

        for key in dataset.keys():
            data_per_context = dataset[key]
        
            est = estimator(data_per_context)
            #Obtaining distributions for biased and unbiased estimators (list of words and a list of their respective probs)
            words_unbiased, probs_unbiased = est.get_estimator_unbiased()
            words_biased, probs_biased = est.get_estimator_biased()
        
            #Expected Calibration Error Data
            ece = ECE(data_per_context)
            human_maj_biased, orig_text_biased = ece.get_ece_data(words_biased, probs_biased)
            conf_acc_human_maj_biased.append( human_maj_biased )
            conf_acc_orig_text_biased.append( orig_text_biased )
                
            human_maj_unbiased, orig_text_unbiased = ece.get_ece_data(words_unbiased, probs_unbiased)
            conf_acc_human_maj_unbiased.append( human_maj_unbiased )
            conf_acc_orig_text_unbiased.append( orig_text_unbiased )

            #Total Variation Distance Data
            tvd = TVD(data_per_context)
            tvd_per_instance_biased = tvd.get_tvd_per_instance_for_biased_est_dist(words_biased, probs_biased)
            tvd_per_instance_unbiased = tvd.get_tvd_per_instance_for_unbiased_est_dist(words_unbiased, probs_unbiased)
            tvd_biased.append(tvd_per_instance_biased)
            tvd_unbiased.append(tvd_per_instance_unbiased)

            #KL-Divergence
            kl_div = KL_div(data_per_context)
            kl_div_biased_per_instance = kl_div.get_kl_div_biased_per_instance(words_biased, probs_biased)
            #kl_div_unbiased_per_instance = kl_div.get_kl_div_unbiased_per_instance(words_unbiased, probs_unbiased)
            kl_div_biased.append(kl_div_biased_per_instance)
            #kl_div_unbiased.append(kl_div_unbiased_per_instance) 
            #we cannot have the kl div. for the unbiased dist. sicne now we are not guaranteed that the q distribution 
            #will be non zero for all values (we added the missing human words which will have prob. 0)

            #Entropy difference
            ent_diff = Entropy(data_per_context)
            ent_diff_biased_per_instance = ent_diff.get_entropy_biased_per_instance(words_biased, probs_biased)
            ent_diff_unbiased_per_instance = ent_diff.get_entropy_unbiased_per_instance(words_unbiased, probs_unbiased)
            entropy_biased.append(ent_diff_biased_per_instance)
            entropy_unbiased.append(ent_diff_unbiased_per_instance)

        
        return conf_acc_human_maj_biased, conf_acc_orig_text_biased, conf_acc_human_maj_unbiased, conf_acc_orig_text_unbiased, tvd_biased, tvd_unbiased, kl_div_biased, entropy_biased, entropy_unbiased

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
    
    def calculate_list_metrics(self, list_met):
        """For a list of instance level metric computations, we compute their key metrics - mean, standard deviation,
        min and max values"""
        array_met = np.array(list_met)
        avg_met = np.mean(array_met)
        std_met = np.std(array_met)
        min_met = np.min(array_met)
        max_met = np.max(array_met)
        return (avg_met, std_met, min_met, max_met)

    def calculate_metrics(self):
        dict_results = {}
        conf_acc_human_maj_biased, conf_acc_orig_text_biased, conf_acc_human_maj_unbiased, conf_acc_orig_text_unbiased, tvd_biased, tvd_unbiased, kl_div_biased, entropy_biased, entropy_unbiased= self.get_metrics_data()
        
        #ECE
        dict_results['ECE_from_biased_dist_human_maj_truth'] = self.calculate_ECE(conf_acc_human_maj_biased)
        dict_results['ECE_from_biased_dist_orig_word_truth'] = self.calculate_ECE(conf_acc_orig_text_biased)
        dict_results['ECE_from_unbiased_dist_human_maj_truth'] = self.calculate_ECE(conf_acc_human_maj_unbiased)
        dict_results['ECE_from_unbiased_dist_orig_word_truth'] = self.calculate_ECE(conf_acc_orig_text_unbiased)
        
        #TVD
        dict_results['TVD_biased_per_instance'] = tvd_biased
        dict_results['TVD_unbiased_per_instance'] = tvd_unbiased

        dict_results['TVD_biased_mean_std_min_max'] = self.calculate_list_metrics(tvd_biased)
        dict_results['TVD_unbiased_mean_std_min_max'] = self.calculate_list_metrics(tvd_unbiased)
        
        #KL-Divergence
        dict_results['KL-Divergenece_biased_per_instance'] = kl_div_biased
        #dict_results['KL-Divergenece_unbiased_per_instance'] = kl_div_unbiased

        dict_results['KL-Divergenece_biased_mean_std_min_max'] = self.calculate_list_metrics(kl_div_biased)
        #dict_results['KL-Divergenece_unbiased_mean_std_min_max'] = self.calculate_list_metrics(kl_div_unbiased)

        #Entropy difference
        dict_results['Entropy_difference_biased_per_instance'] = entropy_biased
        dict_results['Entropy_difference_unbiased_per_instance'] = entropy_unbiased

        dict_results['Entropy_difference_biased_mean_std_min_max'] = self.calculate_list_metrics(entropy_biased)
        dict_results['Entropy_difference_unbiased_mean_std_min_max'] = self.calculate_list_metrics(entropy_unbiased)

        return dict_results

dataset = get_provo_data()
print("Size of dataset:", len(dataset))

# data_loader = data.DataLoader(dataset, batch_size=8, shuffle=False)

metrics_obj = metrics(dataset)        
dict_results = metrics_obj.calculate_metrics()

with open("cal_metrics.json", "w") as outfile:
    json.dump(dict_results, outfile)