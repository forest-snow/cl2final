import numpy as np
import pickle
import random

from collections import Counter
from matplotlib import pyplot as plt
from rnn import *

np.random.seed(3)
mental_health_subreddits = set(['Anger', 'BPD', 'EatingDisorders', 'MMFB', 'StopSelfHarm', 'SuicideWatch', 'addiction', 'alcoholism', 'depression', 'feelgood', 'getting_over_it', 'hardshipmates', 'mentalhealth', 'psychoticreddit', 'ptsd', 'rapecounseling', 'schizophrenia', 'socialanxiety', 'survivorsofabuse', 'traumatoolbox'])

def remove_experts(user_dict, path):
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

def make_corpus(post2vec, user_dict, max_length=30, n_dev=50):
	user_data = list(user_dict.values())
	random.shuffle(user_data)

	train_set, dev_set = [], []
	dev_pos, dev_neg = 0, 0
	train_pos, train_neg = 0, 0
	for user in user_data:
		sequence = []
		# control => dev_set
		if user.label == -1 and dev_neg < n_dev:
			for post in user.posts:
				if post.postid in post2vec:
					vec = post2vec[post.postid]
					sequence.append(vec)
			if len(sequence) > 0:
				dev_neg += 1
				dev_set.append((sequence, 0))
		# sw => dev_set
		elif user.label == 1 and dev_pos < n_dev:
			for post in user.posts:
				if post.postid in post2vec:
					vec = post2vec[post.postid]
					sequence.append(vec)
			if len(sequence) > 0:
				dev_pos += 1
				dev_set.append((sequence, 1))
		# control => train_set
		elif user.label == -1:
			for post in user.posts[-max_length:]:
				if post.postid in post2vec:
					vec = post2vec[post.postid]
					sequence.append(vec)
			if len(sequence) > 0:
				train_neg += 1
				train_set.append((sequence, 0))
		# sw => train_set
		elif user.label == 1 or user.posts[-1].subreddit == 'SuicideWatch':
			for post in user.posts:
				if post.subreddit == 'SuicideWatch' and len(sequence) > 0:
					for i in range(5):
						train_pos += 1
						train_set.append((sequence[-max_length:], 1))
				elif post.postid in post2vec:
					vec = post2vec[post.postid]
					sequence.append(vec)

	print(train_pos, train_neg)
	return train_set, dev_set

def data_batcher(corpus, batch_size):
	data_size = int(len(corpus) / batch_size) * batch_size
	random.shuffle(corpus)

	count = 0
	while count < data_size:
		yield corpus[ count : min(count + batch_size, data_size) ]
		count += batch_size

def predict(model, data, max_length=20):
	predictions = []
	for seq, label in data:
		idx, score = 0, 0
		model.hidden = model.init_hidden()
		while idx < len(seq):
			if idx + 2 * max_length >= len(seq):
				x = Variable(torch.FloatTensor(seq[idx:]))
				score = max(score, model(x).data.numpy()[0])
				break
			else:
				x = Variable(torch.FloatTensor(seq[idx : idx + max_length]))
				score = max(score, model(x).data.numpy()[0])
			idx += max_length

		predictions.append(score)
	return predictions

def test(model, test_set, max_length=20, threshold=0.5):
	labels = [label for seq, label in test_set]
	predictions = predict(model, test_set, max_length=max_length)

	n_rel, n_tp, n_p = [0] * 3
	for label, pred in zip(labels, predictions):
		if label == 1:
			n_rel += 1
			if pred > threshold:
				n_tp += 1
		if pred > threshold:
			n_p += 1

	precision = 0 if n_tp == 0 else n_tp / n_p
	recall = 0 if n_tp == 0 else n_tp / n_rel
	f1 = 2 * precision * recall / (precision + recall)
	print('Precision {}\t Recall {}\t F1 {}'.format(precision, recall, f1))
	return f1

def train(model, train_set, dev_set, num_epochs=20, batch_size=200):
	criterion = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	# keep track of the loss of the model to see if the model converges.
	losses = []
	best_score = 0
	best_model = None
	for epoch in range(num_epochs):
		for batch in data_batcher(train_set, batch_size):
			loss = 0
			optimizer.zero_grad()
			# training
			for seq, label in batch:
				model.hidden = model.init_hidden()
				x = Variable(torch.FloatTensor(seq))
				y = Variable(torch.FloatTensor([label]))
				pred = model(x)
				loss += criterion(pred, y)

			loss /= batch_size
			loss.backward()
			optimizer.step()
			losses.append(loss.data.numpy())
			print(loss.data.numpy())

		# validation
		print('Epoch[{}]'.format(epoch))
		score = test(model, dev_set)
		if score > best_score:
			best_model = model
	return best_model, losses

def extract_vocab(vocab_file, vocab_size=100):
	vocab = []
	word2idx = {}
	with open(vocab_file, 'r') as f:
		for line in f.readlines()[:vocab_size]:
			word = line.strip().split()[0]
			word2idx[word] = len(vocab)
			vocab.append(word)
	return vocab, word2idx

if __name__ == '__main__':
	path = '../umd_reddit_suicidewatch_dataset/'
	pickle_file = '../umd_reddit_suicidewatch_dataset/post_topics'
	model_file = '../umd_reddit_suicidewatch_dataset/rnn_model_1'

	pickle_obj = open(path + 'user_info', 'rb')
	user_dict = pickle.load(pickle_obj)
	crowd_user_dict = remove_experts(user_dict, path)

	pickle_obj = open(pickle_file, 'rb')
	_post2vec = pickle.load(pickle_obj)

	'''
	vocab_size = 100
	vocab, word2idx = extract_vocab(vocab_file, vocab_size=vocab_size)

	post2vec = {}
	for user in crowd_user_dict.values():
		for post in user.posts:
			if post.postid in _post2vec:
				tokens = post.text.split()
				if len(tokens) > 0:
					counter = Counter(tokens)

					vec = np.zeros(vocab_size)
					for word, freq in counter.items():
						if word in word2idx:
							vec[word2idx[word]] = freq
					vec = vec / len(tokens)
					post2vec[post.postid] = np.concatenate((_post2vec[post.postid], vec))
	'''
	train_set, dev_set = make_corpus(_post2vec, crowd_user_dict, n_dev=50)

	model = RNN(100, 10, 1)
	best_model, losses = train(model, train_set, dev_set)
	with open(model_file, 'wb') as f:
		pickle.dump(best_model, f)

	plt.plot(losses)
	plt.show()
