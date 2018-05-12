import pickle
from user import User, Post
from gensim import corpora 
import numpy as np

# call on get_exploratory_data() to get exploratory data


# change paths
PATH_SW = '/Users/myuan/Box Sync/LING_773_Final_Project/umd_reddit_suicidewatch_dataset/'
PATH_USER_INFO = '/Users/myuan/Desktop/user_info'

# number of control users to include
CONTROL = 500

subreddits = ['Anger', 'BPD', 'EatingDisorders', 'MMFB', 'StopSelfHarm', 'SuicideWatch', 'addiction', 'alcoholism', 'depression', 'feelgood', 'getting_over_it', 'hardshipmates', 'mentalhealth', 'psychoticreddit', 'ptsd', 'rapecounseling', 'schizophrenia', 'socialanxiety', 'survivorsofabuse', 'traumatoolbox']


def load_user_dict():
    pickle_obj = open(PATH_USER_INFO, 'rb')
    user_dict = pickle.load(pickle_obj)
    return user_dict

def remove_experts(user_dict):
    # remove any ids that are in expert.csv
    exp_file = PATH_SW + 'reddit_annotation/expert.csv'
    with open(exp_file, 'r') as f:
        data = f.read().splitlines()
    ids = []
    for row in data[1:]:
        entries = row.split(',')
        ids.append(int(entries[0]))

    user_ids = user_dict.keys()
    new_user_dict = dict((i, user_dict[i]) for i in user_ids if i not in ids)
    return new_user_dict

def extract_labeled_users(user_dict, control_users = CONTROL):
    # extract users with label 1 or -1.  
    # control number of control users with label -1 using [control_users]
    new_dict = {}
    n_sw = 0
    n_control = 0
    for user in user_dict:
        label = user_dict[user].label
        if label == 1:
            new_dict[user] = user_dict[user]
            n_sw += 1
        if label == -1 and n_control < control_users:
            new_dict[user] = user_dict[user]
            n_control += 1
    print('SuicideWatch users: {}'.format(n_sw))
    print('Control users: {}'.format(n_control))

    return new_dict




def collect_data(user_dict, avoid=subreddits):
    # collect posts and labels from [user_dict] 
    # avoid is list of subreddits to avoid
    docs = []
    labels = []
    ids = []
    n_posts = 0
    for n_users, (_,user) in enumerate(user_dict.items()):


        # if n_users % 1000 == 0:
        #     print('User {}'.format(n_users))

        for post in user.posts:
        #     if n_posts % 1000 == 0:
        #         print('Post {}'.format(n_posts))
            
            subreddit = post.subreddit

            
            if subreddit not in avoid:    
                text = post.text
                docs.append(text)
                n_posts += 1

                labels.append(user.label)
                ids.append(post.postid)

    print('User {}'.format(n_users))
    print('Post {}'.format(n_posts))
    return docs, labels, ids

def get_exploratory_data():
    user_dict = load_user_dict()
    no_experts_dict = remove_experts(user_dict)
    crowd_user_dict = extract_labeled_users(no_experts_dict)
    docs, labels, _ = collect_data(crowd_user_dict)
    return docs, labels

def filter_words(docs, labels):
    tokens_list = [doc.split() for doc in docs]
    dic = corpora.Dictionary(tokens_list)
    dic.filter_extremes(no_below=20, keep_n=8000, no_above=0.03)

    new_docs = []
    new_labels = []
    for i, tokens in enumerate(tokens_list):
        doc = ''
        for token in tokens:
            if token in dic.token2id:
                doc = doc + token + ' '

        # remove empty posts
        if doc != '':
            new_docs.append(doc)
            new_labels.append(labels[i])
    return new_docs, new_labels, dic

def get_counts(docs, dictionary):
    vocab = list(dictionary.token2id)

    voca_dic = dict()
    voca_list = list()
    for word in vocab:
        voca_dic[word] = len(voca_dic)
        voca_list.append(word)

    doc_ids = list()
    doc_cnt = list()

    for i, doc in enumerate(docs):
        words = set(doc)
        ids = np.array([int(voca_dic[word]) for word in words if word in voca_dic])
        cnt = np.array([int(doc.count(word)) for word in words if word in voca_dic])

        doc_ids.append(ids)
        doc_cnt.append(cnt)

    return np.array(voca_list), doc_ids, doc_cnt

def get_all_data():
    user_dict = load_user_dict()
    # no_experts_dict = remove_experts(user_dict)
    docs, labels, ids = collect_data(user_dict)
    return docs, labels, ids
    
def remove_empty_posts(posts, labels, ids):
    new_posts = []
    new_labels = []
    new_ids = []
    for i, post in enumerate(posts): 
        if len(post) != 0:
            new_posts.append(post)
            new_labels.append(labels[i])
            new_ids.append(ids[i])
    return new_posts, new_labels, new_ids


    