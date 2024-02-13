import numpy as np
import pandas as pd
import torch
from collections import Counter
from gensim.models import word2vec
import gensim
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import gensim.downloader
from sklearn import cluster
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import random
import os
import re
import json
import nltk

def get_estimator(elements):
    """Get the MLE estimate given all words"""
    c = Counter(elements)
    support = list(c.keys())
    counts = list(c.values())
    probs = [count / sum(counts) for count in counts]

    return (support, probs)

def get_common_support(support1, support2):
    return set(support1).union(set(support2)) 

def change_support(old_support, old_probs, new_support):
    """Create new support by adding elements to a support (and their probability value is hence 0)"""
    new_probs = []
    for item in new_support:
        if item in old_support:
            ind = old_support.index(item)
            new_probs.append(old_probs[ind])
        else:
            new_probs.append(0)
    return list(new_support), new_probs

def get_tvd(probs1, probs2):
    tvd = np.sum(np.abs(np.array(probs1) - np.array(probs2)))/2
    return tvd

def get_oracle_elements(words, seed = 0, N = 20):
    """We create two disjoint subsets of the human distribution by sampling without replacement from
    the human distribution (the two disjoing subsets can be comprised by either 10 or 20 samples"""
    random.seed(seed)

    #if the length of the list is odd, we remove one element at random to make the list even,
    #since we want the two disjoint subsets to be of equal length
    if (len(words) % 2 == 1): 
        remove_word = random.sample(words, 1)
        words.remove(remove_word[0])

    #We sample the words that will belong in the first subset and create the second subset by removing
    #from the full word list the ones sampled in the first subset
    subset1 = random.sample(words, N)
    if N == 20:
        subset2 = words.copy()
        for item in subset1:
            subset2.remove(item)
    elif N == 10:
        subset_left = words.copy()
        for item in subset1:
            subset_left.remove(item)
        subset2 = random.sample(subset_left, N)
        
    return subset1, subset2

def missing_word2vec_vectors(samples, probs, vectors):
    word2vec_missing = []
    probs_missing = []
    
    for i, s in enumerate(samples):
        try:
            vectors[s]
        except:
            word2vec_missing.append(s)
            probs_missing.append(probs[i])
    
    return word2vec_missing, probs_missing

def get_word2vec_vectors(samples, vectors):
    X = []
    w = []
    for s in samples:
        X.append(vectors[s])
        w.append(s)
    return X, w 

def cluster_word2vec_vectors(X, w, num_clusters, labels = None, centroids = None, seed = 0):
    if (type(labels) == type(None)) & (type(centroids) == type(None)):
        kmeans = cluster.KMeans(n_clusters = num_clusters, random_state = seed, n_init = 20, max_iter = 400)
        kmeans.fit(np.array(X))

        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

    word_clusters = []
    for i in range(num_clusters):
        word_clusters.append([w[j] for j in range(len(w)) if labels[j] == i])

    return word_clusters, labels, centroids

def cluster_distribution(num_clusters, w, labels, probs):
    """Gathering the probability mass attributed to each semantic cluster"""
    prob_clusters = []
    word_clusters = []
    for i in range(num_clusters):
        prob_cluster = 0
        word_cluster = []
        for j in range(len(labels)):
            if labels[j] == i:
                prob_cluster += probs[j]
                if probs[j] > 0:
                    word_cluster.append(str(w[j]))
        
        prob_clusters.append(prob_cluster)
        word_clusters.append(word_cluster)
    
    return word_clusters, prob_clusters

def pos_tag_distribution(dict_pos_removed):
    """Gathering the probability mass attributed to each pos tag category"""
    probs = []
    for value in dict_pos_removed.values():
        if value == []:
            probs.append(0)
        else:
            probs.append(sum([x[1] for x in value]))
    
    return probs

def combine_distributions(words_clusters, probs_cluster, words_missing, prob_missing):
    """Joining the parts of the full distribution computed (semantic clusters, 
    function word pos tag clusters and unknown words for word2vec) """
    words = words_clusters + words_missing
    probs = probs_cluster + prob_missing
    return words, probs

def normalise_distribution(dist):
    total_mass = sum(dist)
    norm_dist = [item/total_mass for item in dist]

    assert abs(sum(norm_dist) - 1) < 0.001 , 'Invalid distribution'

    return norm_dist

def remove_missing(full_samples, probs_samples, remove_samples):
    for rem in remove_samples:
        ind =  full_samples.index(rem)
        full_samples.remove(rem)
        del probs_samples[ind]
    
    return full_samples, probs_samples

def sse_score(X, centroids, labels):
    curr_sse = 0

    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(X)):
        curr_center = centroids[labels[i]]
        curr_sse += np.linalg.norm(X[i] - curr_center)
    
    return curr_sse
            
def choose_k(X, w, k_max, criterion = 'sse', seed = 0, rel_change = 0.05):
    """Find best k according to a criterion (SSE or SIL)"""
    sil = []
    sse = []
    labels_var_k = []
    centroids_var_k = []
    k_tested = []
    print('k_max', k_max)
    if k_max <= 3:
        step = 1
    else:
        step = k_max//3

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    
    for k in range(2, k_max, step):
        print('k', k)
        k_tested.append(k)
        kmeans = cluster.KMeans(n_clusters = k, random_state = seed)
        kmeans.fit(np.array(X))
        
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        labels_var_k.append(labels)
        centroids_var_k.append(centroids)

        if criterion == 'sil':
            #silhouette_score
            # The silhouette value measures how similar a point is to its own cluster (cohesion) compared to other clusters (separation).
            sil.append(silhouette_score(X, labels, metric = 'euclidean'))

        if criterion == 'sse':
            #Within-Cluster-Sum of Squared Errors: The Squared Error for each point is the square of the distance of the point from 
            # its representation i.e. its predicted cluster center. The WSS score is the sum of these Squared Errors for all the points.
            sse.append(sse_score(X, centroids, labels))

    if criterion == 'sse':
        if len(sse) == 1:
            opt_val = sse[0]
            opt_ind = 0
            opt_k = np.arange(2, k_max, step)[opt_ind]
        else:
            for i in range(1, len(sse)):
                if abs((sse[i] - sse[i-1])/sse[i-1]) < rel_change: #relative change
                    opt_val = sse[i - 1]
                    opt_ind = i - 1
                    opt_k = np.arange(2, k_max, step)[opt_ind]
                    break
                else:
                    opt_ind = sse.index(min(sse))
                    opt_k = np.arange(2, k_max, step)[opt_ind]
        criterion_values = sse
                
    elif criterion == 'sil':
        opt_val = max(sil)
        opt_ind = sil.index(opt_val)
        opt_k = np.arange(2, k_max, step)[opt_ind]
        criterion_values = str(sil)

    
    return opt_k, criterion_values, labels_var_k[opt_ind], centroids_var_k[opt_ind], k_tested

# --------------------- ... ---------------------
# --------------------- ... ---------------------
# --------------------- ... ---------------------


def tvd_semantic(gpt2_samples_pos_tags, human_samples_pos_tags, i):
    #GPT2 Distribution
    words_pos_gpt2, probs_gpt2 = get_estimator([str(tuple(x)).lower() for x in gpt2_samples_pos_tags[i]])
    words_pos_gpt2 = [x.replace("(","").replace(")","").replace("'","").replace(" ","").split(',') for x in words_pos_gpt2]

    word2vec_gpt2_missing, probs_gpt2_missing = missing_word2vec_vectors([x[0] for x in words_pos_gpt2], probs_gpt2, word2vec_vectors)
    words_gpt2, probs_gpt2 = remove_missing([x[0] for x in words_pos_gpt2], probs_gpt2, word2vec_gpt2_missing)

    # --------------------- ... ---------------------
    #Human Distribution
    words_pos_human, probs_human = get_estimator([str(tuple(x)).lower() for x in human_samples_pos_tags[i]])
    words_pos_human = [x.replace("(","").replace(")","").replace("'","").replace(" ","").split(',') for x in words_pos_human]

    word2vec_human_missing, probs_human_missing = missing_word2vec_vectors([x[0] for x in words_pos_human], probs_human, word2vec_vectors)
    words_human, probs_human = remove_missing([x[0] for x in words_pos_human], probs_human, word2vec_human_missing)

    # --------------------- ... ---------------------

    oracle1, oracle2 = get_oracle_elements([item for item in human_samples_pos_tags[i]], seed = 0, N = 20)
    
    #Oracle 1 Distribution
    words_pos_oracle1, probs_oracle1 = get_estimator([str(tuple(x)).lower() for x in oracle1])
    words_pos_oracle1 = [x.replace("(","").replace(")","").replace("'","").replace(" ","").split(',') for x in words_pos_oracle1]
    
    word2vec_oracle1_missing, probs_oracle1_missing = missing_word2vec_vectors([x[0] for x in words_pos_oracle1], probs_oracle1, word2vec_vectors)
    words_oracle1, probs_oracle1 = remove_missing([x[0] for x in words_pos_oracle1], probs_oracle1, word2vec_oracle1_missing)

    # # --------------------- ... ---------------------

    #Oracle 2 Distribution
    words_pos_oracle2, probs_oracle2 = get_estimator([str(tuple(x)).lower() for x in oracle2])
    words_pos_oracle2 = [x.replace("(","").replace(")","").replace("'","").replace(" ","").split(',') for x in words_pos_oracle2]

    word2vec_oracle2_missing, probs_oracle2_missing = missing_word2vec_vectors([x[0] for x in words_pos_oracle2], probs_oracle2, word2vec_vectors)
    words_oracle2, probs_oracle2 = remove_missing([x[0] for x in words_pos_oracle2], probs_oracle2, word2vec_oracle2_missing)

    # # --------------------- ... ---------------------

    samples = list(set(words_gpt2 + words_human)) #No need to add words_oracle1 and words_oracle2, they are included in words_human
    if len(samples) > 1: #We need to have more than 1 word to do clustering
        X, w = get_word2vec_vectors(samples, word2vec_vectors)

        #Standardise vectors
        scl = StandardScaler()
        X = scl.fit_transform(X)

        k_max = len(X) #however words there are to cluster

        if k_max== 2:
            opt_k = 2
            opt_k_labels = None
            opt_k_centroids = None
            k_tested = None
            criterion_value = None
            sse = []
        else:
            opt_k, criterion_value, opt_k_labels, opt_k_centroids, k_tested = choose_k(X, w, k_max, criterion = 'sse')
        
        
        word_clusters, labels, centroids = cluster_word2vec_vectors(X, w, opt_k, labels = opt_k_labels, centroids = opt_k_centroids)
                
        # --------------------- ... ---------------------
        words_human, probs_human = change_support(words_human, probs_human, samples)
        words_gpt2, probs_gpt2 = change_support(words_gpt2, probs_gpt2, samples)

        words_oracle1, probs_oracle1 = change_support(words_oracle1, probs_oracle1, samples)
        words_oracle2, probs_oracle2 = change_support(words_oracle2, probs_oracle2, samples)

        # --------------------- ... ---------------------
        word_human_clusters, dist_human_clusters = cluster_distribution(opt_k, w, labels, probs_human)
        word_gpt2_clusters, dist_gpt2_clusters = cluster_distribution(opt_k, w, labels, probs_gpt2)

        words_final_gpt2, dist_final_gpt2 = combine_distributions(word_gpt2_clusters, dist_gpt2_clusters, word2vec_gpt2_missing, [sum(probs_gpt2_missing)])
        words_final_human, dist_final_human = combine_distributions(word_human_clusters, dist_human_clusters, word2vec_human_missing, [sum(probs_human_missing)])

        #       ------------ ... ------------
        word_oracle1_clusters, dist_oracle1_clusters = cluster_distribution(opt_k, w, labels, probs_oracle1)
        word_oracle2_clusters, dist_oracle2_clusters = cluster_distribution(opt_k, w, labels, probs_oracle2)

        words_final_oracle1, dist_final_oracle1 = combine_distributions(word_oracle1_clusters, dist_oracle1_clusters, word2vec_oracle1_missing, [sum(probs_oracle1_missing)])
        words_final_oracle2, dist_final_oracle2 = combine_distributions(word_oracle2_clusters, dist_oracle2_clusters, word2vec_oracle2_missing, [sum(probs_oracle2_missing)])

        #       ------------ ... ------------
        results = [word_clusters, criterion_value, k_tested, int(opt_k)]    
        results_oracles = [get_tvd(dist_final_oracle1, dist_final_oracle2), (words_final_oracle1, dist_final_oracle1), (words_final_oracle2, dist_final_oracle2)]
        results_model_human = [get_tvd(dist_final_gpt2, dist_final_human), (words_final_gpt2, dist_final_gpt2), (words_final_human, dist_final_human)]
    else:
        results = [] 
        results_oracles = []
        results_model_human = []

    return results, results_oracles, results_model_human


def get_tvd_semantic(gpt2_samples_pos_tags, human_samples_pos_tags, lower_bound, upper_bound):
    dict_results = {}
    for i in range(lower_bound, upper_bound + 1):
        dict_results[i] = {}
        dict_results[i]['general'], dict_results[i]['tvd_sem_oracle1_oracle2'], dict_results[i]['tvd_sem_human_model'] = tvd_semantic(gpt2_samples_pos_tags, human_samples_pos_tags, i)
    
    return dict_results

# --------------------- ... ---------------------
# --------------------- ... ---------------------
# --------------------- ... ---------------------

#How to obtain pos_tags_info_universal.json

# def get_provo_data(input_data):
#     """A function that takes all .json files we created with info for the Provo Corpus
#     and merges it into one dictionary"""
    
#     #We merge all information in one dictionary
#     # Each data point corresponds to all the information relevant to us for a given context in Provo Corpus
#     joint_dict = {}
    
#     count = 0
#     for filename in input_data:
#         f = open(filename)
#         data = json.load(f)
#         f.close()

#         for text_id in data.keys():
#             if (int(text_id) > 0) & (int(text_id) <= 55):
#                 for word_num in data[text_id].keys():
#                     joint_dict[count] = data[text_id][word_num]
#                     joint_dict[count]['original_positioning'] = {'text_id':text_id, 'word_num':word_num}
                
#                     count = count + 1

#     return joint_dict

# input_data = ['Paragraphs-1-1.json', 'Paragraphs-2-2.json', 'Paragraphs-3-3.json',
#     'Paragraphs-4-4.json', 'Paragraphs-5-9.json', 'Paragraphs-10-14.json', 
#     'Paragraphs-15-19.json', 'Paragraphs-20-24.json', 'Paragraphs-25-29.json',
#     'Paragraphs-30-34.json', 'Paragraphs-35-39.json', 'Paragraphs-40-44.json', 
#     'Paragraphs-45-47.json', 'Paragraphs-48-50.json', 'Paragraphs-51-53.json',
#      'Paragraphs-54-55.json']

# d = get_provo_data(input_data)

# human_samples = []
# human_samples_pos_tags = []

# original_corpus_words = []
# gpt2_samples = []
# gpt2_samples_pos_tags = []

# contexts = []

# for key in d.keys():
#     context = d[key]['context']['text']
#     contexts.append(context)
#     human_samp= [[x['pred']]*int(x['count']) for x in d[key]['human']]
#     human_samp = [str(item) for sublist in human_samp for item in sublist]
#     human_pos_tags = [nltk.pos_tag(nltk.word_tokenize(context.lower() + ' ' + sample.lower()), tagset='universal')[-1] for sample in human_samp]

#     original_word = d[key]['original']['pred']

#     gpt2_samp = [x for x in d[key]['ancestral_samples'] if x != 'Failed to generate word']
#     gpt2_samp = [[x['pred']]*int(x['count']) for x in gpt2_samp]
#     gpt2_samp = [str(item) for sublist in gpt2_samp for item in sublist]
#     gpt2_pos_tags = [nltk.pos_tag(nltk.word_tokenize(context.lower() + ' ' + sample.lower()), tagset='universal')[-1] for sample in gpt2_samp]

#     human_samples.append(human_samp)
#     gpt2_samples.append(gpt2_samp)
#     original_corpus_words.append([original_word])
#     human_samples_pos_tags.append(human_pos_tags)
#     gpt2_samples_pos_tags.append(gpt2_pos_tags)

# info = {'human_samples' : human_samples,
#         'human_samples_pos_tags' : human_samples_pos_tags,
#         'original_corpus_words' : original_corpus_words,
#         'gpt2_samples' : gpt2_samples,
#         'gpt2_samples_pos_tags' : gpt2_samples_pos_tags}

# import json

# with open("pos_tags_info_universal.json", "w") as fp:
#     json.dump(info, fp)  

f = open('pos_tags_info_universal.json')

info = json.load(f)

human_samples = info['human_samples'] 
human_samples_pos_tags = info['human_samples_pos_tags']
original_corpus_words = info['original_corpus_words']
gpt2_samples = info['gpt2_samples']
gpt2_samples_pos_tags = info['gpt2_samples_pos_tags']

word2vec_vectors = gensim.downloader.load('word2vec-google-news-300')

lower_bound = 0
upper_bound = 2686
dict_results = get_tvd_semantic(gpt2_samples_pos_tags, human_samples_pos_tags, lower_bound, upper_bound)

with open("test"+ str(lower_bound) + '-' + str(upper_bound) + ".json", "w") as final:
    json.dump(dict_results, final)






