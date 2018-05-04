from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import re

wnl = WordNetLemmatizer()
def lemmatize_pos(sent):
    tokens = []
    for word, tag in pos_tag(word_tokenize(sent)):
        wntag = tag[0].lower()

        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
        if not wntag:
            lemma = word
        else:
            lemma = wnl.lemmatize(word, wntag)
        tokens.append(lemma)
    return tokens

def preprocess(text):
    no_dash_text = re.sub(r'-', ' ', text)
    tokens = no_dash_text.split()
    lowercase_tokens = [token.lower() for token in tokens]
    english_tokens = [re.sub(r'[^a-z]', '', token) for token in lowercase_tokens]
    sent = ' '.join(english_tokens) 
    lemma_tokens = lemmatize_pos(sent)
    long_tokens = [token for token in lemma_tokens if len(token) >= 3]
    return ' '.join(long_tokens)