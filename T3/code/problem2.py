# CS 181, Harvard University
# Spring 2016
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
import random
from Perceptron import Perceptron
from sklearn.svm import SVC

# Implement this class
class KernelPerceptron(Perceptron):
	def __init__(self, numsamples):
		self.numsamples = numsamples

	def sanity_check(self, X, Y):
		leq_pos = 0
		leq_neg = 0
		g_pos = 0
		g_neg = 0
		for i in range(len(X)):
			if X[i][0] <= X[i][1]:
				if Y[i] > 0 :
					leq_pos += 1
				else:
					leq_neg += 1
			else:
				if Y[i] > 0:
					g_pos += 1
				else: 
					g_neg += 1
		print "leq_pos", leq_pos
		print "leq_neg", leq_neg
		print "g_pos", g_pos
		print "g_neg", g_neg

	def K(self, x_t, x_i):
		total = 0
		for j in range(len(x_t)):
			total += x_t[j]*x_i[j]
		return total
	# Implement this!
	def fit(self, X, Y):
		self.X = X
		self.Y = Y
		self.S = dict()
		self.alphas = dict()
		# self.S_count = 0
		n = len(X)
		for itr in range(self.numsamples):
			t = random.randint(0,n-1)
			y_hat = 0
			# print "len", len(self.S)
			for i,x_i in self.S.iteritems():
				temp1 = self.alphas[i]
				temp2 = X[t]
				# print len(X)
				# print "n", n			
				y_hat += self.alphas[i] * self.K(X[t], x_i) # b=0 for now
			if (Y[t]*y_hat <= 0):
				self.S[t] = X[t]				
				self.alphas[t] = Y[t]
				# self.S_count += 1
		# print self.S_count

	def validate(self, X_val, Y_val):
		Y_pred = self.predict(X_val)
		correct = 0
		wrong = 0
		for i in range(len(Y_pred)):
			if Y_pred[i] == Y_val[i]:
				correct += 1
			else:
				wrong += 1
		return (float(correct))/(float(correct+wrong))

	# Implement this!
	def predict(self, X):
		Y = list()
		for t in range(len(X)):
			y_hat = 0
			for i,x_i in self.S.iteritems():
				y_hat += self.alphas[i]*self.K(X[t], x_i) # b=0 for now
			if (y_hat>0):
				Y.append(1)
			else:
				Y.append(-1)

		return np.array(Y)



# Implement this class
class BudgetKernelPerceptron(Perceptron):
	def __init__(self, beta, N, numsamples):
		self.beta = beta
		self.N = N
		self.numsamples = numsamples
		
	# Implement this!
	def fit(self, X, Y):
		self.X = X
		self.Y = Y
		self.S = dict() # map t to (x1,x2)s
		self.alphas = dict() # map t to alpha_t
		n = len(X)
		for itr in range(self.numsamples):
			t = random.randint(0,n-1)
			y_hat = 0
			for i,x_i in self.S.iteritems():			
				y_hat += self.alphas[i] * self.K(X[t], x_i) # b=0 for now
			if (Y[t]*y_hat <= self.beta):
				self.S[t] = X[t]
				self.alphas[t] = Y[t]
				if len(self.S) > self.N:
					key_max = max(self.S, key=lambda key: Y[key]*(y_hat-self.alphas[key]*self.K(X[key], X[key])))
					self.S.pop(key_max)
					self.alphas.pop(key_max)
	def K(self, x_t, x_i):
		total = 0
		for j in range(len(x_t)):
			total += x_t[j]*x_i[j]
		return total

	def validate(self, X_val, Y_val):
		Y_pred = self.predict(X_val)
		correct = 0
		wrong = 0
		for i in range(len(Y_pred)):
			if Y_pred[i] == Y_val[i]:
				correct += 1
			else:
				wrong += 1
		return (float(correct))/(float(correct+wrong))

	# Implement this!
	def predict(self, X):
		Y = list()
		for t in range(len(X)):
			y_hat = 0
			for i,x_i in self.S.iteritems():
				y_hat += self.alphas[i]*self.K(X[t], x_i) # b=0 for now			
			if (y_hat>0):
				Y.append(1)
			else:
				Y.append(-1)
		return np.array(Y)



# Do not change these three lines.
data = np.loadtxt("data.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]

val = np.loadtxt("val.csv", delimiter=',')
X_val = data[:, :2]
Y_val = data[:, 2]

"""
clf = SVC()
clf.fit(X,Y)
correct = 0
wrong = 0
for i in range(len(X_val)):
	y_pred = clf.predict(Y_val[i])
	if y_pred == Y[i]:
		correct += 1
	else:
		wrong += 1
print "Svc", (float(correct)/float(correct+wrong))
"""

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
beta = 0.05
N = 100
numsamples = 20000 # 20000

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
k = KernelPerceptron(numsamples)
k.sanity_check(X,Y)
k.fit(X,Y)
k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)
print "validation", k.validate(X_val, Y_val)

bk = BudgetKernelPerceptron(beta, N, numsamples)
bk.fit(X, Y)
bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)
print "validation", bk.validate(X_val, Y_val)