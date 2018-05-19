import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

class RNN(nn.Module):
	def __init__(self, input_size, hidden_dim, output_size):
		super(RNN, self).__init__()

		self.hidden_dim = hidden_dim
		self.hidden = self.init_hidden()
		self.lstm = nn.GRU(input_size, hidden_dim)
		self.h2o = nn.Linear(hidden_dim, output_size)
		self.sigmoid = nn.Sigmoid()

	def forward(self, sequence):
		x = sequence.view(len(sequence), 1, -1)
		lstm_out, self.hidden = self.lstm(x, self.hidden)
		output = self.sigmoid(self.h2o(lstm_out[-1]))
		return output

	def init_hidden(self):
		return Variable(torch.zeros(1, 1, self.hidden_dim))
