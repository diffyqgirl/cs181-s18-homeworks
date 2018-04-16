import csv
import math
import matplotlib.pyplot as plt
import numpy as np

x = []
z = []

with open("kf-data.csv", 'r') as fh:
	reader = csv.reader(fh)
	skip = True
	for row in reader:
		# print row
		if(not skip):
			z.append(float(row[1]))
			x.append(float(row[2]))
		skip = False
s_e = 0.05
s_g = 1
s_p = 1
mu_p = 5
mu_t = list()
mu_t.append(mu_p)
s_t = list()
s_t.append(s_p)
K = []
for t in range(len(x)):
	K.append((s_e**2 + s_p**2)/(s_e**2+s_g**2+s_p**2))
	s_p = s_g*math.sqrt(K[t])
	mu_p = mu_p + K[t]*(x[t]-mu_p)
	mu_t.append(mu_p)
	s_t.append(s_p)
print "K", K[len(x)-1]
print "mu_t", mu_t[len(x)]
print "s_t", s_t[len(x)]

print "true mean: ", sum(z)/float(len(z))
print "true std. deviation: ", np.std(z)

fig = plt.figure(1)
plt.errorbar(range(0, len(x) + 1), mu_t, yerr=[2*x for x in s_t])
plt.xlabel('t')
plt.ylabel('mu_t')
plt.show()
	