import extract

from ptm import GibbsSupervisedLDA
from ptm.nltk_corpus import get_ids_cnt
from ptm.utils import convert_cnt_to_list, get_top_words
from ptm.slda_vb import sLDA


N_TOPICS = 100

def print_topics(word_topic_matrix, vocab, eta):
    for ti in eta.argsort():
        top_words = get_top_words(word_topic_matrix, vocab, ti, n_words=10)
        print('Eta', eta[ti], 'Topic', ti ,':\t', ','.join(top_words))

def slda_gibbs(corpus, labels, n_docs, n_vocab, n_topics, vocab):
    r_var = 0.01

    model = GibbsSupervisedLDA(n_docs, n_vocab, n_topics, sigma=r_var)
    model.fit(corpus, labels)

    print_topics(model.TW, vocab, model.eta)

    return model

def slda_vb(corpus, labels, n_docs, n_vocab, n_topics, vocab):
    model = sLDA(corpus, labels, vocab, n_topics, alpha=1/N_TOPICS, sigma=1)
    model.fit(max_iter=10)

    print_topics(model.beta, vocab, model.eta)

    return model

def check_empty(corpus):
    for doc in corpus:
        if len(doc) == 0:
            print('empty doc')


if __name__ == '__main__':
    
    docs, labels = extract.get_exploratory_data()





    docs, labels, dictionary = extract.filter_words(docs, labels)

    print('\n\nloaded data')
    print('{} docs'.format(len(docs)))
    tokens = [doc.split() for doc in docs]

    vocab, word_ids, word_cnt = extract.get_counts(tokens, dictionary)
    # print('vocab')
    # print(vocab)
    # print('word_ids')
    # print(word_ids)
    # print('word_cnt')
    # print(word_cnt)


    print('{} words'.format(len(vocab)))
    corpus = convert_cnt_to_list(word_ids, word_cnt)

    n_docs = len(corpus)
    n_vocab = vocab.size
    check_empty(corpus)

    slda_vb(corpus, labels, n_docs, n_vocab, N_TOPICS, vocab)


