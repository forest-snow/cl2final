#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 14:39:20 2018

@author: Richburg
"""

import sys
import argparse
from collections import Counter
from nltk.corpus import stopwords
import time
from collections import OrderedDict
import math
import pickle
#from user import User, Post
from gensim import corpora, models
#import gensim

subreddits = ['Anger', 'BPD', 'EatingDisorders',
    'MMFB', 'StopSelfHarm', 'SuicideWatch', 'addiction', 'alcoholism', 'depression', 'feelgood', 'getting_over_it', 'hardshipmates',
    'mentalhealth', 'psychoticreddit', 'ptsd', 'rapecounseling', 'schizophrenia', 'socialanxiety', 'survivorsofabuse',
    'traumatoolbox']

def remove_experts(user_dict):
    exp_file = path + 'reddit_annotation/expert.csv'
    with open(exp_file, 'r') as f:
        data = f.read().splitlines()
    ids = []
    for row in data[1:]:
        entries = row.split(',')
        ids.append(int(entries[0]))

    user_ids = user_dict.keys()
    new_user_dict = dict((i, user_dict[i]) for i in user_ids if i not in ids)
    return new_user_dict

def split_data(user_dict):
    #exp_file = path + 'reddit_posts/sw_users/split_80-10-10/TRAIN.txt'
    split_file = path + 'reddit_annotation/crowd.csv'
    with open(split_file, 'r') as f:
        data = f.read().splitlines()
    ids_1 = []
    ids_2 = []
    for row in data[1:]:
        entries = row.split(',')
        user_id = int(entries[0])
        user_label = int(entries[1])
        #annotated suicide watch data
        if user_label > 0:
            ids_1.append(user_id)
        #other data to compare too (<0 for control, == 0 unannotated data)     
        elif user_label < 0:
            ids_2.append(user_id)
    user_ids = user_dict.keys()
    
    #with open(exp_file, 'r') as k:
      #  want = k.read().split()
        
    #ids_1 = [uid for uid in ids_1 if uid in want]
    #ids_2 = [uid for uid in ids_2 if uid in want]
    user_dict_new1 = dict((i, user_dict[i]) for i in user_ids if i in ids_1)
    user_dict_new2 = dict((i, user_dict[i]) for i in user_ids if i in ids_2)
    return user_dict_new1, user_dict_new2

def collect_posts(user_dict, avoid=subreddits):
    # collect_posts from [user_dict] 
    # avoid is list of subreddits to avoid
    docs = []
    author2doc = {}
    n_posts = 0
    for n_users, (_,user) in enumerate(user_dict.items()):

        if n_users % 1000 == 0:
            print('User {}'.format(n_users))

        for post in user.posts:
            if n_posts % 1000 == 0:
                print('Post {}'.format(n_posts))
            try:
                subreddit = post.reddit
                print('has subreddit')
            except AttributeError:
                subreddit = ''
            
            if subreddit not in avoid:    
                text = post.text
                docs.append(text)

                #append_dct(author2doc, user.label, [n_posts])
                n_posts += 1

    print('User {}'.format(n_users))
    print('Post {}'.format(n_posts))
    return docs, author2doc

def uni_prob_smth(text,vocab_size):
    count = Counter(text)
    L = float(vocab_size)
    prob_dict = {}
    #k = 1/ float(L)
    k = 1
    for word in text:
        prob_word = (count[word] + k) / (L*k + float(L))
        prob_dict.update({word: prob_word})
    return prob_dict

def kl_div(distr1, distr2, vocab_size):
    total_sum = 0
    kl_div_dict = {}
    for word in distr1:
        prob_word1 = distr1[word]
        if word not in distr2:
            prob_word2 = 1 / float(vocab_size)
        else:
            prob_word2 = distr2[word]
            
        Log = math.log(prob_word1 / prob_word2 , 2)
        result = float(prob_word1 * Log)
        kl_div_dict.update({word: result})
        total_sum += result
    kl_div_dict_sort = OrderedDict(sorted(kl_div_dict.items(), key = lambda item: item[1], reverse = True))
    return total_sum , kl_div_dict_sort

def create_bows(docs_tokens):
    dictionary = corpora.Dictionary(docs_tokens)
    dictionary.filter_extremes(no_below=30, keep_n=5000, no_above=0.04)
    bows = [dictionary.doc2bow(doc) for doc in docs_tokens]
    #print('{} words'.format(len(dictionary.token2id)))
    return bows, dictionary

def remove_stopwords(text):
    new_list = []
    stop_words = set(stopwords.words('english'))
    for r in text:
        if r not in stop_words:
            new_list.append(r)
    return new_list

def get_collocations(docs):
    # get collocations for documents
    # returns documents but with bigrams (unigrams that are separated by _)
    tokens = [doc.split() for doc in docs]
    phrases = models.phrases.Phrases(tokens, min_count=100, threshold=10000.0, max_vocab_size=1000)
    bigrams = [sentence.split() for sentence in phrases[docs]]
    return bigrams

if __name__ == '__main__':
    #argparser = argparse.ArgumentParser()
    #argparser.add_argument('--dist_1',type=str, required=True,
	#						help="Data for distribution 1")
    #argparser.add_argument('--dist_2', type=str, required=True,
	#						help="Data for distribution 2")
    
    path = '/Users/Richburg/Dropbox/SPRING_18/LING_773/Assignments/FINAL/umd_reddit_suicidewatch_dataset/'
    #args = argparser.parse_args()
    sub_path = path + '/reddit_annotation'

    pickle_obj = open(path+'test_users', 'rb')
    user_dict = pickle.load(pickle_obj)
    print(len(user_dict))
    crowd_user_dict = remove_experts(user_dict)
    print(len(crowd_user_dict))
    
    sw_user_dict, con_user_dict = split_data(crowd_user_dict)
    
    docs_sw = collect_posts(sw_user_dict)
    docs_con = collect_posts(con_user_dict)
    
    col=0
    
    if col ==1:
        docs_sw = [get_collocations(docs) for sublist in docs_sw for docs in sublist]
        docs_con = [get_collocations(docs) for sublist in docs_con for docs in sublist]
    
    #doc_tokens_sw = [doc.split() for doc in docs_sw]
    #doc_tokens_con = [doc.split() for doc in docs_con]
    
    print(docs_sw[0])
    new_docs_sw = [item for sublist in docs_sw for item in sublist]
    new_docs_con = [item for sublist in docs_con for item in sublist]
    print(new_docs_sw[0])
    #new_docs_sw = [item for sublist in new_docs_sw for item in sublist]
    #new_docs_con = [item for sublist in new_docs_con for item in sublist]
    print(new_docs_sw[0])
    
    flat_docs_sw = [word for doc in new_docs_sw for word in doc.split()]
    flat_docs_con = [word for doc in new_docs_con for word in doc.split()]
    
    #bows_sw, dict_sw = create_bows(doc_tokens_sw)
    #bows_con, dict_con = create_bows(doc_tokens_con)
    
    #new_docs_sw = []
    #new_docs_con = []
    #for word in flat_docs_sw:
     #   if word in dict_sw:
      #      new_docs_sw.append(word)
            
    #for word in flat_docs_con:
     #   if word in dict_con:
      #      new_docs_con.append(word)
    
    flat_docs_sw = remove_stopwords(flat_docs_sw)
    flat_docs_con = remove_stopwords(flat_docs_con)
    
    total_voc = flat_docs_sw + flat_docs_con
    vocab_size = len(Counter(total_voc))
    print(vocab_size)
    prob_sw = uni_prob_smth(flat_docs_sw,vocab_size)
    prob_con = uni_prob_smth(flat_docs_con,vocab_size)
    
    div, div_dict = kl_div(prob_sw, prob_con,vocab_size)
    div2, div_dict2 = kl_div(prob_con, prob_sw,vocab_size)
    print(len(div_dict))
    print(len(div_dict2))
    
    top_number = 400
    div_file2 = 'top_div_unantoan_' + str(top_number)+ '.txt'
    with open(div_file2, 'w') as f2:
        f2.write('D(p||q) = ' + str(div2) + '\n')
        runthru2 = list(div_dict2)
        if top_number <= len(runthru2):
            M = top_number
        else:
            M = len(runthru2)
        for i in range(0,M):
            word = runthru2[i]
            div2 = float(div_dict2[word])
            #row = [word, div]
            f2.write(str(word) + ' ' + str(div2) + '\n')
    
    div_file = 'top_div_antounan_' + str(top_number)+ '.txt'
    with open(div_file, 'w') as f:
        f.write('D(p||q) = ' + str(div) + '\n')
        runthru1 = list(div_dict)
        if top_number <= len(runthru1):
            M = top_number
        else:
            M = len(runthru1)
        for i in range(0,M):
            word = runthru1[i]
            div = float(div_dict[word])
            #row = [word, div]
            f.write(str(word) + ' ' + str(div) + '\n')
    