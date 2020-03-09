import sys
from scipy import optimize
import numpy as np
import math
import itertools
import scipy.stats as stats
import scipy.integrate as integrate
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-whitegrid')
mpl.style.use('seaborn')
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
import ipywidgets as widgets
from ipywidgets import interact, interact_manual


logger = logging.getLogger(__name__)


# Static functions drawing samples:
def draw_gh(n,variance):
	temp = stats.truncnorm.rvs(0.5,2,scale=np.sqrt(variance),size=n)
	return temp/sum(temp)

def draw_ph(n,loc,variance):
	return np.random.normal(loc,np.sqrt(variance),n)

def draw_gh_ph(n,loc,variance_p,variance_g,corr):
	gh = draw_gh(n,variance_g)
	gh_var = np.var(gh)
	beta_ = corr*variance_p
	mu = loc-beta_*np.mean(gh)
	var_epsilon = variance_p-beta_**2*gh_var
	ph = mu+beta_*gh+np.random.normal(0,np.sqrt(var_epsilon),n)
	return ph,gh


class Logit:
	"""
	Set up model with logit-like demand function.
	"""

	def __init__(self,name="",**kwargs):
		self.name = name
		self.base_par()
		self.upd_par(kwargs)

	def base_par(self):
		self.I = 1000
		self.alpha = 0.5
		self.sigma = 10
		self.upper = 2
		self.lower = 0
		self.eps = 0 # increase numerical stability of unbounded functions, e.g. log().

		self.n=1760
		self.loc=200
		self.variance_p=100
		self.variance_g=10
		self.corr=0
		self.utils()

	def utils(self):
		self.u_cge = lambda C,E: C**(self.alpha)*E**(1-self.alpha)
		if hasattr(self, 'npv'):
			self.uhour = lambda E: Logit.staticmethod_uh(self.sigma,self.lower,self.upper,E,self.npv['Eh_l'],self.npv['Eh_u'])
			self.uh = lambda E: self.uhour(E)-self.uhour(self.npv['Eh'])
			self.u = lambda C,E: self.u_cge(C,sum(E))+self.uh(E)
			self.utilde = lambda lambda_: np.exp(self.sigma*lambda_*(self.ph-self.npv['p']))
			self.grad_utilde = lambda lambda_: self.sigma*(self.ph-self.npv['p'])*self.utilde(lambda_)
			self.chi = lambda utilde: self.lower+((1-self.lower)*(self.upper-self.lower))/(1-self.lower+(self.upper-1)*utilde)
			self.grad_chi = lambda utilde: (1-self.lower)*(self.upper-self.lower)*((1-self.lower+(self.upper-1)*utilde))**(-2)*(self.upper-1)


	def upd_par(self,kwargs):
		for key,value in kwargs.items():
			setattr(self,key,value)
		self.utils()

	@staticmethod
	def staticmethod_uh(sigma,l,u,E,El,Eu):
		return (1/sigma)*sum(E*np.log((1-l)/(u-1))-(E-El)*np.log(E-El)-(Eu-E)*np.log(Eu-E))

	def solve_npv(self):
		self.npv = {'C': self.alpha*self.I,
					'p': sum(self.gh*self.ph)}
		self.npv['E'] = (1-self.alpha)*self.I/self.npv['p']
		self.npv['lambda'] = self.alpha*(self.npv['C']/self.npv['E'])**(self.alpha-1)
		self.npv['Eh'] = self.gh*self.npv['E']
		self.npv['Eh_l'] = self.npv['Eh']*self.lower-self.eps
		self.npv['Eh_u'] = self.npv['Eh']*self.upper+self.eps
		self.npv['uhour']= Logit.staticmethod_uh(self.sigma,self.lower,self.upper,self.npv['Eh'],self.npv['Eh_l'],self.npv['Eh_u'])
		self.npv['uscale'] = -self.npv['uhour']
		self.utils()
		self.npv['u'] = self.u(self.npv['C'],self.npv['Eh'])

	def solve_fp(self):
		if not hasattr(self,'npv'):
			self.solve_npv()
		def f(x): # x[0] = lambda, x[1] = C, x[2:] = Eh
			f = np.empty(len(x))
			f[0] = x[0]-self.alpha*(x[1]/(sum(x[2:])))**(self.alpha-1) # FOC for C
			f[1] = self.I-x[1]-sum(self.ph*x[2:]) # Budget
			f[2:] = x[2:]-self.npv['Eh']*self.chi(self.utilde(x[0])) # Allocation on hours
			return f
		def grad(x):
			g = np.zeros([self.n+2,self.n+2])
			g[0,0] 	= 1
			g[0,1] 	= self.alpha*(self.alpha-1)*x[1]**(self.alpha-2)*(sum(x[2:])**(1-self.alpha))
			g[0,2:] = self.alpha*(1-self.alpha)*x[1]**(self.alpha-1)*(sum(x[2:])**(-self.alpha))
			g[1,0] 	= 0
			g[1,1] 	= -1
			g[1,2:]	= -self.ph
			g[2:,0] = self.gh*self.npv['E']*self.grad_chi(self.utilde(x[0]))*self.grad_utilde(x[0])
			g[2:,2:]= np.diag(np.ones(self.n))
			return g
		x,info,ier,msg = optimize.fsolve(f,np.append([self.npv['lambda'],self.npv['C']],self.npv['Eh']),fprime=grad,full_output=True)
		print(msg)
		self.fp = {}
		self.fp['lambda'],self.fp['C'],self.fp['Eh'] = Logit.x2args(x)
		self.fp['E'] = sum(self.fp['Eh'])
		self.fp['p'] = sum(self.ph*self.fp['Eh'])/self.fp['E']
		self.fp['Eh_l'] = self.fp['Eh']*self.lower-self.eps
		self.fp['Eh_u'] = self.fp['Eh']*self.upper+self.eps
		self.fp['uhour'] = self.uhour(self.fp['Eh'])
		self.fp['uscale'] = self.npv['uscale']
		self.fp['u'] = self.u(self.fp['C'],self.fp['Eh'])

	@staticmethod
	def x2args(x):
		return x[0],x[1],x[2:]

	# @staticmethod
	# def args2x(sol):
	# 	return np.append([self.npv['lambda'],self.npv['C']],self.npv['Eh'])

	# Sample functions:
	def draw_g(self,ph=None):
		if ph:
			self.gh = draw_gh(len(ph),self.variance_g)
		else:
			self.gh = draw_gh(self.n,self.variance_g)

	def draw_p(self,gh=None):
		if gh:
			self.ph = draw_ph(len(gh),self.loc,self.variance_p)
		else:
			self.ph = draw_ph(self.n,self.loc,self.variance_p)

	def sample(self):
		self.ph,self.gh = draw_gh_ph(self.n,self.loc,self.variance_p,self.variance_g,self.corr)