import pickle
from user import User, Post
from gensim import corpora, models


# Change these variables
N_TOPICS = 20
PATH_SW = '/Users/myuan/Box Sync/LING_773_Final_Project/umd_reddit_suicidewatch_dataset/'
PATH_USER_INFO = '/Users/myuan/Desktop/user_info'



subreddits = ['Anger', 'BPD', 'EatingDisorders',
    'MMFB', 'StopSelfHarm', 'SuicideWatch', 'addiction', 'alcoholism', 'depression', 'feelgood', 'getting over it', 'hardshipmates',
    'mentalhealth', 'psychoticreddit', 'ptsd', 'rapecounseling', 'schizophrenia', 'socialanxiety', 'survivorsofabuse',
    'traumatoolbox']


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

def extract_labeled_users(user_dict, n_users = 2000):
    # extract users with label 1 or -1.  
    # control number of users with label -1 using [n_users]
    new_dict = {}
    n_sw = 0
    n_control = 0
    for user in user_dict:
        label = user_dict[user].label
        if label == 1:
            new_dict[user] = user_dict[user]
            n_sw += 1
        if label == -1 and n_control < n_users:
            new_dict[user] = user_dict[user]
            n_control += 1
    print('SuicideWatch users: {}'.format(n_sw))
    print('Control users: {}'.format(n_control))

    return new_dict

def append_dct(dct, key, value):
    # helper function for collect posts
    if key not in dct:
        dct[key] = value
    else:
        dct[key] = dct[key] + value


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

                append_dct(author2doc, user.label, [n_posts])
                n_posts += 1

    print('User {}'.format(n_users))
    print('Post {}'.format(n_posts))
    return docs, author2doc

def get_collocations(docs):
    # get collocations for documents
    # returns documents but with bigrams (unigrams that are separated by _)
    tokens = [doc.split() for doc in docs]
    phrases = models.phrases.Phrases(tokens, min_count=100, threshold=10000.0, max_vocab_size=1000)
    bigrams = [sentence.split() for sentence in phrases[docs]]
    return bigrams
    # print(phrases.vocab)


def nonempty_docs(bows):
    # helper function for create_bows
    # removes any empty docs, which can create errors for author-topic model
    doc_ids = []
    for i, bow in enumerate(bows):
        if bow != []:
            doc_ids.append(i)
    return doc_ids

def create_bows(docs_tokens, author2doc):
    # create bag-of-words from tokens 
    # need to update author2doc if there are any empty documents
    dictionary = corpora.Dictionary(docs_tokens)
    dictionary.filter_extremes(no_below=30, keep_n=5000, no_above=0.07)
    bows = [dictionary.doc2bow(doc) for doc in docs_tokens]
    print('{} words'.format(len(dictionary.token2id)))
    ids = nonempty_docs(bows)
    nonempty_bows = [bows[i] for i in ids]
    author2doc[1] = [doc for doc in author2doc[1] if doc in ids]
    author2doc[-1] = [doc for doc in author2doc[-1] if doc in ids]

    return bows, dictionary, author2doc

def top_topic_words(model, dictionary):
    # return top 10 words for each topic
    topic_words = []
    for i in range(model.num_topics):
        tt = model.get_topic_terms(i,10)
        topic_words.append([dictionary[pair[0]] for pair in tt])
    return topic_words

def train_lda_model(bows, dictionary, n_topics):
    # train lda model
    # return model and top topic words
    print('training topic model')
    model = models.LdaModel(bows, num_topics=n_topics)
    print('finished training')
    topic_words = top_topic_words(model, dictionary)
    return model, topic_words

def train_at_model(bows, dictionary, author2doc, n_topics):
    # train at model
    # return model and top topic words
    model = \
        models.atmodel.AuthorTopicModel(
            corpus = bows, 
            num_topics = n_topics, 
            author2doc=author2doc, 
            id2word=dictionary
            )
    topic_words = top_topic_words(model,dictionary)
    return model, topic_words 

def output_topics(model, topic_words, label):
    # output topics for author-topic model
    # model[i] contains top topics for documents labeled i
    file = 'at_topics/topics_'+str(model.num_topics)+'_'+str(label)+'.txt'
    with open(file, 'w') as f:
        for topic, prob in model[label]:
            f.write('Topic {}\n'.format(topic))
            f.write(' '.join(topic_words[topic])+'\n')
            f.write(str(prob)+'\n')

if __name__ == '__main__':

    pickle_obj = open(PATH_USER_INFO, 'rb')

    # getting the right users
    user_dict = pickle.load(pickle_obj)
    print(len(user_dict))
    no_experts_dict = remove_experts(user_dict)
    crowd_user_dict = extract_labeled_users(no_experts_dict)
    print(len(crowd_user_dict))

    # collecting posts
    docs, author2doc = collect_posts(crowd_user_dict)

    # creating tokens
    docs_tokens = [doc.split() for doc in docs]

    # creating bag-of-words
    bows, dictionary, author2doc = create_bows(docs_tokens, author2doc)
    print('number of documents {}'.format(len(bows)))

    # model, topic_words = \
    #     train_lda_model(bows, dictionary, models.atmodel.AuthorTopicModel, 10)


    model, topic_words = train_at_model(bows, dictionary, author2doc, N_TOPICS)

    print(topic_words)
    output_topics(model, topic_words, 1)
    output_topics(model, topic_words, -1)

    # topic_file = 'at_topics/topics_'+str(model.num_topics)+'.txt'
    # with open(topic_file, 'w') as f:
    #     for row in topic_words:
    #         print(row)
    #         f.write(' '.join(row) + '\n')

