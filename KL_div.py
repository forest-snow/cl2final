#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 14:39:20 2018

@author: Richburg
"""

from collections import Counter
from nltk.corpus import stopwords
from collections import OrderedDict
import math
from nltk import ngrams
import pickle

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

#P is first data, Q is second data (given as annotated, unannotated or control)
def split_data(user_dict,P,Q):
    split_file = path + 'reddit_annotation/crowd.csv'
    with open(split_file, 'r') as f:
        data = f.read().splitlines()
    ids_1 = []
    ids_2 = []
    for row in data[1:]:
        entries = row.split(',')
        user_id = int(entries[0])
        user_label = int(entries[1])
        if user_label == P:
            ids_1.append(user_id)    
        elif user_label == Q:
            ids_2.append(user_id)
    user_ids = user_dict.keys()

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
    k = 1/ float(L)
    #k = 1
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

def remove_stopwords(text):
    new_list = []
    stop_words = set(stopwords.words('english'))
    for r in text:
        if r not in stop_words:
            new_list.append(r)
    return new_list

def make_grams(lst, n):
    grams = list(ngrams(lst,n))
    return grams

def help_make_title(P_or_Q):
    if P_or_Q == 1:
        output = str('annotated')
    elif P_or_Q == 0:
        output = str('unannotated')
    elif P_or_Q == -1:
        output = str('control')
    return output

if __name__ == '__main__':
    #Replace path with directory that contains test_users preprocessed dataset
    path = '/Users/Richburg/Dropbox/SPRING_18/LING_773/Assignments/FINAL/umd_reddit_suicidewatch_dataset/'
    sub_path = path + '/reddit_annotation'

    pickle_obj = open(path+'test_users', 'rb')
    user_dict = pickle.load(pickle_obj)
    print(len(user_dict))
    crowd_user_dict = remove_experts(user_dict)
    print(len(crowd_user_dict))
    
    #P and Q represent dataset distributions; 
    #1 is annnotated SW, 0 is unannotated SW, -1 is control
    P=1
    Q=0
    P_user_dict, Q_user_dict = split_data(crowd_user_dict,P,Q)
    
    docs_P = collect_posts(P_user_dict)
    docs_Q = collect_posts(Q_user_dict)
    
    new_docs_P = [item for sublist in docs_P for item in sublist]
    new_docs_Q = [item for sublist in docs_Q for item in sublist]
    
    #Change n for the size of ngrams
    n = 2
    if n >= 2:
        flat_docs_P = []
        for docu in new_docs_P:
            words = docu.split()
            flat_docs_P.append(words)
            for k in range(n,n+2):
                grams = make_grams(words,k)
                flat_docs_P.append(grams)
        
        flat_docs_Q = []
        for docu in new_docs_Q:
            words = docu.split()
            flat_docs_Q.append(words)
            for k in range(n,n+2):
                grams = make_grams(words,k)
                flat_docs_Q.append(grams)
            
        flat_docs_P = [word for doc in flat_docs_P for word in doc]
        flat_docs_Q = [word for doc in flat_docs_Q for word in doc]
    else:
        flat_docs_P = [word for doc in new_docs_P for word in doc.split()]
        flat_docs_Q = [word for doc in new_docs_Q for word in doc.split()]

    total_voc = flat_docs_P + flat_docs_Q
    vocab_size = len(Counter(total_voc))
    print(vocab_size)
    prob_P = uni_prob_smth(flat_docs_P,vocab_size)
    prob_Q = uni_prob_smth(flat_docs_Q,vocab_size)
    
    div, div_dict = kl_div(prob_P, prob_Q,vocab_size)
    div2, div_dict2 = kl_div(prob_Q, prob_P,vocab_size)
    print(len(div_dict))
    print(len(div_dict2))
    
    #max number of words contributing to KL divergence
    top_number = 600
    title_P = help_make_title(P)
    title_Q = help_make_title(Q)
    
    #div_file2 = 'top_' + str(top_number) +'_div_' + title_Q + '_to_' +  title_P + '_' + str(n)+ 'grams' +'.txt'
    #with open(div_file2, 'w') as f2:
     #   f2.write('D(p||q) = ' + str(div2) + '\n')
      #  runthru2 = list(div_dict2)
       # if top_number <= len(runthru2):
        #    M = top_number
        #else:
        #    M = len(runthru2)
        #for i in range(0,M):
         #   word = runthru2[i]
          #  div2 = float(div_dict2[word])
           # f2.write(str(word) + ' ' + str(div2) + '\n')
    
    div_file = 'top_' + str(top_number) +'_div_' + title_P + '_to_' +  title_Q + '_' + str(n)+ 'grams' +'.txt'
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
            f.write(str(word) + ' ' + str(div) + '\n')
    