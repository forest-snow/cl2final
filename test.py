import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import Counter, defaultdict
from torch.autograd import Variable
from matplotlib import pyplot as plt
from rnn import *
from sklearn import metrics
from train import *

mental_health_subreddits = set(['Anger', 'BPD', 'EatingDisorders', 'MMFB', 'StopSelfHarm', 'SuicideWatch', 'addiction', 'alcoholism', 'depression', 'feelgood', 'getting_over_it', 'hardshipmates', 'mentalhealth', 'psychoticreddit', 'ptsd', 'rapecounseling', 'schizophrenia', 'socialanxiety', 'survivorsofabuse', 'traumatoolbox'])
dining_subreddits = set(['electronic_cigarette', 'PipeTobacco', 'Drugs', 'HIFW'])
finance_subreddits = set(['personalfinance', 'theydidthemath'])
emotion_subreddits = set(['relationships', 'Showerthoughts', 'offmychest', 'AskWomen'])

def get_experts(user_dict, path):
	exp_file = path + 'reddit_annotation/expert.csv'
	with open(exp_file, 'r') as f:
		data = f.read().splitlines()
	user_labels = {}
	for row in data[1:]:
		entries = row.split(',')
		user_labels[int(entries[0])] = entries[2]

	user_ids = user_dict.keys()
	new_user_dict = dict((i, user_dict[i]) for i in user_ids if i in user_labels)
	return new_user_dict, user_labels


def make_testset(post2vec, user_dict, user_labels):
	data, users, labels = [], [], []
	for user in user_dict.values():
		sequence = []
		if user.label == -1:
			for post in user.posts:
				if post.postid in post2vec:
					vec = post2vec[post.postid]
					sequence.append(vec)
			
			data.append((sequence, 0))
			users.append(user.userid)
			labels.append(user_labels[user.userid])

		elif user.label == 1:
			for post in user.posts:
				if post.postid in post2vec:
					vec = post2vec[post.postid]
					sequence.append(vec)

			data.append((sequence, 1))
			users.append(user.userid)
			labels.append(user_labels[user.userid])

	return data, users, labels


def get_scores(model, sequence):
	scores = []
	hidden_states = defaultdict(list)
	for step in range(1, len(sequence) + 1):
		model.hidden = model.init_hidden()
		x = Variable(torch.FloatTensor(sequence[:step]))
		scores.append(model(x).data.numpy()[0])
		hidden = model.hidden.view(-1)
		for idx in range(model.hidden_dim):
			hidden_states[idx].append(hidden[idx])
	return scores, hidden_states


def inspect(model, user, sequence, posts, mental_posts, max_length=40):
	timeline = [post.timestamp for post in posts]
	turning_points = [post.timestamp for post in mental_posts]
	if len(sequence) <= max_length:
		scores, hidden_states = get_scores(model, sequence)
		if len(scores) >= 15:
			topics = []
			# plt.plot(timeline, scores, '-.', label='output')
			for idx in [1, 7, 8]:
				plt.plot(timeline, hidden_states[idx], '-.', label='h_%d' % idx)
			for t in turning_points:
				plt.axvline(x=t, color='deeppink')
			for i, vec in enumerate(sequence):
				topic = vec.argmax()
				topics.append(topic)
			print(topics)
			plt.title('User: ' + str(user))
			plt.legend(loc="lower right")
			plt.xlabel('timeline')
			plt.ylabel('model probability')
			plt.show()


if __name__ == '__main__':
	path = '../umd_reddit_suicidewatch_dataset/'
	pickle_file = '../umd_reddit_suicidewatch_dataset/post_topics'
	model_file = '../umd_reddit_suicidewatch_dataset/rnn_model_1'

	pickle_obj = open(path + 'user_info', 'rb')
	user_dict = pickle.load(pickle_obj)
	expert_user_dict, user_labels = get_experts(user_dict, path)

	pickle_obj = open(pickle_file, 'rb')
	_post2vec = pickle.load(pickle_obj)

	test_set, userids, labels = make_testset(_post2vec, expert_user_dict, user_labels)
	model = pickle.load(open(model_file, 'rb'))
	predictions = predict(model, test_set, max_length=20)
	true_labels = [label for _, label in test_set]

	'''
	fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions)
	auc_roc = metrics.auc(fpr, tpr)
	plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc_roc)
	plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
	plt.title('Receiver operating characteristic curve')
	plt.legend(loc="lower right")
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.show()
	'''

	c, d, tc, td = [0] * 4
	for data, userid, label, pred in zip(test_set, userids, labels, predictions):
		if label == 'c':
			c += 1
			if pred >= 0.5:
				tc += 1
				mental_posts = [post for post in user_dict[userid].posts if post.subreddit in emotion_subreddits]
				posts = [post for post in user_dict[userid].posts if post.postid in _post2vec]
				inspect(model, userid, data[0], posts, mental_posts)
		elif label == 'd':
			d += 1
			if pred >= 0.5:
				td += 1
				mental_posts = [post for post in user_dict[userid].posts if post.subreddit in emotion_subreddits]
				posts = [post for post in user_dict[userid].posts if post.postid in _post2vec]
				inspect(model, userid, data[0], posts, mental_posts)

	print('C {}/{}\t D {}/{}\t'.format(tc, c, td, d))
