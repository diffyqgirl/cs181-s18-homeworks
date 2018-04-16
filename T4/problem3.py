# CS 181, Spring 2017
# Homework 4: Clustering
# Name:
# Email:

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class KMeans(object):
	# K is the K in KMeans
        def __init__(self, K):
		self.K = K

	# X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
	def fit(self, X):
		self.X = X
		self.means = list()
		N = len(X)

		# Forgy method of initialization
		for i in range(self.K):
			self.means.append(X[np.random.randint(0,N-1)])
		# print "means", self.means

		max_itrs = 1000
		itr = 0
		clusters = list()# list of cluster indices
		self.dists = list()
		while(itr < max_itrs): # todo make this stop prematurely if necessary
			is_change = False
			total_dist = 0
			for i in range(N):
				idx = 0
				dist = np.inf
				for j in range(self.K):
					temp = np.linalg.norm(np.subtract(X[i], self.means[j]))
					if temp < dist:
						dist = temp
						idx = j
				total_dist += dist
				if (itr == 0):
					clusters.append(idx)
				else:
					if (clusters[i] != idx):
						is_change = True
					clusters[i] = idx
			self.dists.append(total_dist)
			if (not is_change) and itr > 0:
				break
			# recompute means
			totals = [np.zeros(X[0].shape) for i in range(self.K)]
			counts = [0 for i in range(self.K)]
			for i in range(N):
				totals[clusters[i]] = np.add(totals[clusters[i]], X[i])
				counts[clusters[i]] += 1
			for j in range(self.K):
				self.means[j] = np.divide(totals[j], counts[j])

			itr += 1
		self.clusters = clusters
		print "It took ", itr, "iterations to find a local optimum"


	def get_dists(self):
		return self.dists

	# This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
	def get_mean_images(self):
		return self.means

	# This should return the arrays for D images from each cluster that are representative of the clusters.
	def get_representative_images(self, D):
		imgs = [[] for i in range(self.K)]
		vals = [[] for i in range(self.K)]
		for i in range(len(self.X)):
			cluster = self.clusters[i]
			if (len(imgs[cluster]) < D):
				imgs[cluster].append(self.X[i])
				vals[cluster].append(np.linalg.norm(np.subtract(self.X[i], self.means[cluster])))
			else:
				max_idx = 0
				for j in range(D):
					if vals[cluster][j] > vals[cluster][max_idx]:
						max_idx = j
				temp = np.linalg.norm(np.subtract(self.X[i], self.means[cluster]))
				if vals[cluster][max_idx] > temp:
					vals[cluster][max_idx] = temp
					imgs[cluster][max_idx] = self.X[i]
		return imgs
				#temp1 = np.subtract(imgs[cluster][D-1])
				# if np.linalg.norm(imgs[cluster][D-1]) < np.linalg.norm(X[i])

	# img_array should be a 2D (square) numpy array.
	# Note, you are welcome to change this function (including its arguments and return values) to suit your needs. 
	# However, we do ask that any images in your writeup be grayscale images, just as in this example.
	def create_image_from_array(self, img_array):
		plt.figure()
		plt.imshow(img_array, cmap='Greys_r')
		plt.show()
		return

# This line loads the images for you. Don't change it! 
pics = np.load("images.npy", allow_pickle=False)

# You are welcome to change anything below this line. This is just an example of how your code may look.
# That being said, keep in mind that you should not change the constructor for the KMeans class, 
# though you may add more public methods for things like the visualization if you want.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.

K = 15
# KMeansClassifier = KMeans(K=10, useKMeansPP=False)
KMeansClassifier = KMeans(K=K)
KMeansClassifier.fit(pics)
means = KMeansClassifier.get_mean_images()


# KMeansClassifier.create_image_from_array(np.vstack(means))
reps = KMeansClassifier.get_representative_images(3)
reps_img = list() #np.hstack(means)
for i in range(K):
	reps_img.append(np.vstack(reps[i]))
KMeansClassifier.create_image_from_array(np.vstack((np.hstack(means),np.hstack(reps_img))))
# for k in range(K):
	# KMeansClassifier.create_image_from_array(means[k])
	# reps.append(KMeansClassifier.get_representative_images(3))
# KMeansClassifier.create_image_from_array(np.hstack(KMeansClassifier.get_representative_images()))
# KMeansClassifier.create_image_from_array(means[0])
# KMeansClassifier.create_image_from_array(reps[0][0])

# dists = KMeansClassifier.get_dists()
# print "k means objective: ", dists[len(dists)-1]
# plt.plot([i for i in range(len(dists))], dists)
# plt.show()
# KMeansClassifier.create_image_from_array(pics[0])




