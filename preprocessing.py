import argparse
import csv
import os
import re
import _pickle as pickle

from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from user import *

wnl = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))

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
	long_tokens = [token for token in lemma_tokens if len(token) >= 3 and token not in stopWords]
	return ' '.join(long_tokens)

def extract_files(path, pattern):
	file_list = []
	for root, dirs, files in os.walk(path):
		for file in files:
			if file.endswith(pattern):
				file_list.append(os.path.join(root, file))

	return file_list

def load_posts(user_dict, path, pattern, in_control=False):
	for file in extract_files(path, pattern):
		with open(file, 'r') as f:
			cnt=0
			line = f.readline()
			while line:
				cnt+=1
				if cnt % 100 == 0:
					print(cnt)
				entries = line.strip().split('\t')
				if len(entries) <= 3:
					continue

				postid = entries[0].strip()
				subreddit = entries[3].strip()
				userid = int(entries[1])
				timestamp = int(entries[2])
				if len(entries) > 5:
					text = ' '.join([entries[3], entries[4], entries[5]])
				else:
					text = ' '.join([entries[3], entries[4]])

				text = preprocess(text)
				post = Post(postid, subreddit, text, timestamp)

				if userid not in user_dict:
					if in_control:
						user_dict[userid] = User(userid, -1)
					else:
						user_dict[userid] = User(userid, -2)

				user_dict[userid].add_post(post)

				line = f.readline()

def load_users(path, pattern):
	user_dict = dict()
	for file in extract_files(path, pattern):
		with open(file, 'r') as f:
			reader = csv.DictReader(f)
			for row in reader:
				userid = int(row['user_id'])
				user_dict[userid] = User(userid, int(row['label']))
	return user_dict

if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--post_dir', type=str, required=True,
							help="Directory of posts.")
	argparser.add_argument('--user_dir', type=str, required=True,
							help="Directory of user annotations.")
	argparser.add_argument('--output_file', type=str, required=True,
							help="Path to the output file.")

	args = argparser.parse_args()
	user_dict = load_users(args.user_dir, 'csv')
	load_posts(user_dict, os.path.join(args.post_dir, 'sw_users'), 'posts', in_control=False)
	load_posts(user_dict, os.path.join(args.post_dir, 'controls'), 'posts', in_control=True)
	print(len(user_dict))

	with open(args.output_file, 'wb') as f:
		pickle.dump(user_dict, f)
