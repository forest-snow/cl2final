import pickle
from user import User, Post
from gensim import corpora, models

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

def collect_posts(user_dict):
    docs = []

    n_posts = 0
    for n_users, (_,user) in enumerate(user_dict.items()):
        if n_users % 1000 == 0:
            print('User {}'.format(n_users))

        for post in user.posts:
            if n_posts % 1000 == 0:
                print('Post {}'.format(n_posts))
            text = post.text
            docs.append(text)
            n_posts += 1

    print('User {}'.format(n_users))
    print('Post {}'.format(n_posts))
    return docs

def get_collocations(docs):
    tokens = [doc.split() for doc in docs]
    phrases = models.phrases.Phrases(tokens, min_count=100, threshold=10000.0, max_vocab_size=1000)
    bigrams = [sentence.split() for sentence in phrases[docs]]
    return bigrams
    # print(phrases.vocab)

def create_bows(docs_tokens):
    dictionary = corpora.Dictionary(docs_tokens)
    dictionary.filter_extremes(no_below=30, keep_n=5000, no_above=0.05)
    bows = [dictionary.doc2bow(doc) for doc in docs_tokens]
    # print(dictionary.token2id)
    return bows, dictionary

def top_topic_words(lda, dictionary):
    topic_words = []
    for i in range(lda.num_topics):
        tt = lda.get_topic_terms(i,10)
        topic_words.append([dictionary[pair[0]] for pair in tt])
    return topic_words


if __name__ == '__main__':
    path = '/Users/myuan/Box Sync/LING_773_Final_Project/umd_reddit_suicidewatch_dataset/'

    pickle_obj = open(path+'test_users', 'rb')
    user_dict = pickle.load(pickle_obj)
    print(len(user_dict))
    crowd_user_dict = remove_experts(user_dict)
    print(len(crowd_user_dict))

    docs = collect_posts(crowd_user_dict)

    docs_tokens = [doc.split() for doc in docs]

    bows, dictionary = create_bows(docs_tokens)

    print('training topic model')
    lda = models.LdaModel(bows, num_topics=50)
    print('finished training')
    topic_words = top_topic_words(lda, dictionary)

    topic_file = 'topics/topics_'+str(lda.num_topics)+'.txt'
    with open(topic_file, 'w') as f:
        for row in topic_words:
            print(row)
            f.write(' '.join(row) + '\n')