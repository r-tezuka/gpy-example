from scipy.io import loadmat
import scipy.io as spio
import GPy
import matplotlib.pyplot as plt

d = spio.loadmat('input/3Class.mat')
X = d['DataTrn']
X -= X.mean(0)
L = d['DataTrnLbls'].nonzero()[1]
input_dim = 2 # How many latent dimensions to use

kernel = GPy.kern.RBF(input_dim, ARD=True) + GPy.kern.Bias(input_dim) + GPy.kern.Linear(input_dim) + GPy.kern.White(input_dim)
model = GPy.models.BayesianGPLVM(X, input_dim, kernel=kernel, num_inducing=30)
model.optimize(messages=True, max_iters=5e3)
model.plot_latent(labels=L)
plt.savefig('output/fig8.png')