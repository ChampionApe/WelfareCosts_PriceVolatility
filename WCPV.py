from scipy import optimize
import numpy as np
import pandas as pd
import math
import scipy.stats as stats
import scipy.integrate as integrate
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-whitegrid')
mpl.style.use('seaborn')
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
plt.rcParams['font.family'] = 'Palatino Linotype'

logger = logging.getLogger(__name__)


# Static functions drawing samples:
def draw_gh(n,variance,seed=None):
	if seed:
		np.random.seed(seed)
	temp = stats.truncnorm.rvs(0.5,2,scale=np.sqrt(variance),size=n)
	return temp/sum(temp)

def draw_ph(n,loc,variance_p,gh,corr,seed=None):
	if seed:
		np.random.seed(seed)
	gh_std = np.sqrt(gh)
	beta_f = lambda var_p: corr*np.sqrt(var_p)/gh_std
	eps_std_f= lambda var_p: np.sqrt(var_p-beta_f(var_p)**2*gh_std**2)
	epsh_f = lambda var_p: np.random.normal(0,eps_std_f(var_p),n)
	ph_f = lambda var_p: loc+beta_f(var_p)*(gh-1/n)+epsh_f(var_p)
	eps_std = eps_std_f(variance_p)
	epsh = epsh_f(variance_p)
	ph = ph_f(variance_p)
	return ph,epsh,gh_std,eps_std

def draw_gh_ph(n,loc,variance_p,variance_g,corr,seed=None):
	if seed:
		np.random.seed(seed)
	gh = draw_gh(n,variance_g,seed)
	gh_std = np.sqrt(np.var(gh))
	beta_f = lambda var_p: corr*np.sqrt(var_p)/gh_std
	eps_std_f= lambda var_p: np.sqrt(var_p-beta_f(var_p)**2*gh_std**2)
	epsh_f = lambda var_p: np.random.normal(0,eps_std_f(var_p),n)
	ph_f = lambda var_p: loc+beta_f(var_p)*(gh-1/n)+epsh_f(var_p)
	eps_std = eps_std_f(variance_p)
	epsh = epsh_f(variance_p)
	ph = ph_f(variance_p)
	return ph,gh,epsh,gh_std,eps_std

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
		self.sigma = 1
		self.upper = 2
		self.lower = 0
		self.eps = 0 # increase (>=0) for numerical stability of unbounded functions, e.g. log().

		self.sample='ph_gh' # Allowed elements: 'ph','gh','ph_gh', None.
		self.type='Logit' # allowed elements: Logit, MNL, GNMNL.
		self.BP_Ref = 'level' # allowed elements: level, share. In Logit case, only 'level' is allowed. 
		self.n=500
		self.loc=200
		self.variance_p=1
		self.variance_g=1
		self.corr=0
		self.seed=30
		self.utils()

	def utils(self):
		self.u_cge = lambda C,E: C**(self.alpha)*E**(1-self.alpha)
		if hasattr(self, 'npv'):
			if self.type=='Logit':
				self.f_uh = lambda Eh,El,Eu: (1/self.sigma)*(Eh*np.log((1-self.lower)/(self.upper-1))-(Eh-El)*np.log(Eh-El)-(Eu-Eh)*np.log(Eu-Eh))
				self.f_uh_sum = lambda Eh: sum(self.f_uh(Eh,self.npv['Eh_l'],self.npv['Eh_u']))
				self.f_uh_all = lambda Eh: self.f_uh_sum(Eh)+self.npv['uscale']
				self.f_u = lambda C,Eh: self.u_cge(C,sum(Eh))+self.f_uh_all(Eh)
				# Auxiliary functions:
				self.utilde = lambda lambda_: np.exp(self.sigma*lambda_*(self.ph-self.npv['p']))
				self.grad_utilde = lambda lambda_: self.sigma*(self.ph-self.npv['p'])*self.utilde(lambda_)
				self.chi = lambda utilde: self.lower+((1-self.lower)*(self.upper-self.lower))/(1-self.lower+(self.upper-1)*utilde)
				self.grad_chi = lambda utilde: (1-self.lower)*(self.upper-self.lower)*((1-self.lower+(self.upper-1)*utilde))**(-2)*(self.upper-1)
				self.eh_demand = lambda lambda_,Eh: self.npv['Eh']*self.chi(self.utilde(lambda_))
			elif self.type=='MNL':
				self.utilde = lambda lambda_: np.exp(self.sigma*lambda_*(self.pmin-self.ph))/(sum(self.gh*np.exp(self.sigma*lambda_*(self.pmin-self.ph))))
				if self.BP_Ref=='level':
					self.f_uh = lambda Eh,gh: (1/self.sigma)*(Eh*np.log(gh*self.npv['E'])-Eh*(np.log(Eh)-1))
					self.f_uh_sum = lambda Eh: sum(self.f_uh(Eh,self.gh))
					self.f_uh_all = lambda Eh: self.f_uh_sum(Eh)+self.npv['uscale']
					self.eh_demand = lambda lambda_,Eh: self.npv['Eh']*self.utilde(lambda_)
				elif self.BP_Ref=='share':
					self.f_uh_h = lambda Eh,gh: (1/self.sigma)*Eh*(np.log(gh/Eh))
					self.f_uh_t = lambda Eh: (1/self.sigma)*sum(Eh)*np.log(sum(Eh))
					self.f_uh_sum = lambda Eh: self.f_uh_t(Eh)+sum(self.f_uh_h(Eh,self.gh))
					self.f_uh_all = lambda Eh: self.f_uh_sum(Eh)+self.npv['uscale']
					self.eh_demand = lambda lambda_,Eh: self.gh*sum(Eh)*self.utilde(lambda_)
			self.f_u = lambda C,Eh: self.u_cge(C,sum(Eh))+self.f_uh_all(Eh)


	def upd_par(self,kwargs):
		for key,value in kwargs.items():
			setattr(self,key,value)
		self.utils()

	def upd_all(self,kwargs):
		self.upd_par(kwargs)
		self.draw_sample()
		self.solve_npv()
		self.solve_fp()

	def solve_npv(self):
		self.npv = {'C': self.alpha*self.I,
					'p': sum(self.gh*self.ph)}
		self.npv['E'] = (1-self.alpha)*self.I/self.npv['p']
		self.npv['lambda'] = self.alpha*(self.npv['C']/self.npv['E'])**(self.alpha-1)
		self.npv['Eh'] = self.gh*self.npv['E']
		self.npv['Eh_l'] = self.npv['Eh']*self.lower-self.eps
		self.npv['Eh_u'] = self.npv['Eh']*self.upper+self.eps
		self.utils()
		self.npv['uhour']= self.f_uh_sum(self.npv['Eh']) 
		self.npv['uscale'] = -self.npv['uhour']
		self.npv['u'] = self.f_u(self.npv['C'],self.npv['Eh'])

	def solve_fp(self):
		if not hasattr(self,'npv'):
			self.solve_npv()
		def f(x):
			f = np.empty(len(x))
			f[0] = x[0]-self.alpha*(x[1]/(sum(x[2:])))**(self.alpha-1)
			f[1] = self.I-x[1]-sum(self.ph*x[2:]) # Budget
			f[2:] = x[2:]-self.eh_demand(x[0],x[2:]) # Allocation on hours
			return f
		# def grad(x):
		# 	g = np.zeros([self.n+2,self.n+2])
		# 	g[0,0] 	= 1
		# 	g[0,1] 	= self.alpha*(self.alpha-1)*x[1]**(self.alpha-2)*(sum(x[2:])**(1-self.alpha))
		# 	g[0,2:] = self.alpha*(1-self.alpha)*x[1]**(self.alpha-1)*(sum(x[2:])**(-self.alpha))
		# 	g[1,0] 	= 0
		# 	g[1,1] 	= -1
		# 	g[1,2:]	= -self.ph
		# 	g[2:,0] = self.gh*self.npv['E']*self.grad_chi(self.utilde(x[0]))*self.grad_utilde(x[0])
		# 	g[2:,2:]= np.diag(np.ones(self.n))
		# 	return g
		x,info,ier,msg = optimize.fsolve(f,np.append([self.npv['lambda'],self.npv['C']],self.npv['Eh']),full_output=True)
		print(msg)
		self.fp = {}
		self.fp['lambda'],self.fp['C'],self.fp['Eh'] = Logit.x2args(x)
		self.fp['E'] = sum(self.fp['Eh'])
		self.fp['p'] = sum(self.ph*self.fp['Eh'])/self.fp['E']
		self.fp['Eh_l'] = self.npv['Eh']*self.lower-self.eps
		self.fp['Eh_u'] = self.npv['Eh']*self.upper+self.eps
		self.fp['uhour'] = self.f_uh_sum(self.fp['Eh'])
		self.fp['uscale'] = self.npv['uscale']
		self.fp['u'] = self.f_u(self.fp['C'],self.fp['Eh'])

	@staticmethod
	def x2args(x):
		return x[0],x[1],x[2:]



	#################################################################################################
	#									 PLOTTING PROPERTIES:										#
	#################################################################################################

	def plots(self,plots,titles=None,return_fig=None):
		"""
		Wrapper around plot properties. Plots should be a list of plotting properties. 
		"""
		if isinstance(plots,str):
			fig,axes = plt.subplots(1,1,figsize=(6,4))
			plt.subplot(1,1,1)
			if titles:
				eval('self.'+plots+'({a})'.format(a=titles))
			else:
				eval('self.'+plots+'()')
		else:
			if not titles:
				titles = []
			if len(plots)==1:
				fig,axes = plt.subplots(1,1,figsize=(6,4))
				plt.subplot(1,1,1)
				if plots[0] in titles:
					eval('self.'+plots[0]+'({a})'.format(a=titles[plots[0]]))
				else:
					eval('self.'+plots[0]+'()')
			else:
				fig,axes = plt.subplots(round(len(plots)/2),2,figsize=(12,4*round(len(plots)/2)))
				for i in range(len(plots)):
					plt.subplot(round(len(plots)/2),2,i+1)
					if plots[i] in titles:
						eval('self.'+plots[i]+'({a})'.format(a=titles[plots[i]]))
					else:
						eval('self.'+plots[i]+'()')
		fig.tight_layout()
		if return_fig:
			return fig,axes

	def plt_compare_eh(self,add_title='standard'):
		plt.plot(pd.Series(self.npv['Eh']).sort_values().reset_index(drop=True))
		plt.plot(pd.Series(self.fp['Eh']).sort_values().reset_index(drop=True))
		plt.xlabel('Hours of the year (sorted)', fontsize=13)
		plt.ylabel('$E_h$', fontsize=13)
		plt.legend(('$E_h$, constant prices', '$E_h$, flexible prices'), fontsize=13)
		if add_title=='standard':
			plt.title('Sorted hourly consumption with/without flexible prices', fontweight='bold',fontsize=16)
		elif add_title:
			plt.title(add_title,fontsize=13)

	def plt_ph(self, add_title='standard'):
		plt.plot(pd.Series(self.ph).sort_values().reset_index(drop=True))
		plt.xlabel('Hours of the year (sorted)', fontsize=13)
		plt.ylabel(('Hourly prices'), fontsize=13)
		if add_title=='standard':
			plt.title('Sorted hourly prices', fontweight='bold',fontsize=16)
		elif add_title:
			plt.title(add_title,fontsize=13)

	def plt_gh(self, add_title='standard'):
		plt.plot(pd.Series(self.gh).sort_values().reset_index(drop=True))
		plt.xlabel('Hours of the year (sorted)', fontsize=13)
		plt.ylabel(('Preferences $g_h$'), fontsize=13)
		if add_title=='standard':
			plt.title('Sorted hourly preferences', fontweight='bold',fontsize=16)
		elif add_title:
			plt.title(add_title,fontsize=13)

	def plt_eh_npv_gh(self, add_title='standard'):
		plt.scatter(self.npv['Eh'],self.gh)
		plt.xlabel('$E_h$ without flexible prices',fontsize=13)
		plt.ylabel('Preferences $g_h$',fontsize=13)
		if add_title=='standard':
			plt.title('Hourly consumption/preferences (npv)', fontweight='bold', fontsize=16)
		elif add_title:
			plt.title(add_title,fontsize=13)

	def plt_eh_npv_ph(self, add_title='standard'):
		plt.scatter(self.npv['Eh'],self.ph)
		plt.xlabel('$E_h$ without flexible prices',fontsize=13)
		plt.ylabel('Hourly prices $p_h$',fontsize=13)
		if add_title=='standard':
			plt.title('Hourly consumption/prices (npv)', fontweight='bold', fontsize=16)
		elif add_title:
			plt.title(add_title,fontsize=13)

	def plt_eh_fp_gh(self,add_title='standard'):
		plt.scatter(self.fp['Eh'],self.gh)
		plt.xlabel('$E_h$, flexible prices',fontsize=13)
		plt.ylabel('Preferences $g_h$',fontsize=13)
		if add_title=='standard':
			plt.title('Hourly consumption/preferences (fp)', fontweight='bold', fontsize=16)
		elif add_title:
			plt.title(add_title,fontsize=13)

	def plt_eh_fp_ph(self,add_title='standard'):
		plt.scatter(self.fp['Eh'],self.ph)
		plt.xlabel('$E_h$, flexible prices',fontsize=13)
		plt.ylabel('Hourly prices $p_h$',fontsize=13)
		if add_title=='standard':
			plt.title('Hourly consumption/prices (fp)', fontweight='bold', fontsize=16)
		elif add_title:
			plt.title(add_title,fontsize=13)

	def plt_uhour(self,add_title='standard'):
		ehgrid = np.linspace(self.npv['Eh_l'][0]+0.000000000001,self.npv['Eh_u'][0]-0.0000001,100)
		if self.type=='Logit':
			uh = (1/self.sigma)*(ehgrid * np.log((1-self.lower)/(self.upper-1))-(ehgrid-self.npv['Eh_l'][0])*np.log(ehgrid-self.npv['Eh_l'][0])-(self.npv['Eh_u'][0]-ehgrid)*np.log(self.npv['Eh_u'][0]-ehgrid))
		if self.type=='MNL':
			if self.BP_Ref=='level':
				uh = self.f_uh(ehgrid,self.gh[0])
			elif self.BP_Ref=='share':
				Etgrid = np.array([self.npv['E']-self.npv['Eh'][0]+ehgrid[i] for i in range(len(ehgrid))])
				uh = self.f_uh_h(ehgrid,self.gh[0])+(1/self.sigma)*Etgrid*np.log(Etgrid)
		plt.plot(ehgrid,uh)
		plt.axvline(x=self.gh[0]*self.npv['E'], color=colors[1], linestyle='--', label='$g_hE_t^*$')
		plt.xlabel('$E_h$',fontsize=13)
		plt.ylabel('$u^{hour}_h$',fontsize=13)
		plt.legend(('$u^{hour}_h$', '$g_hE_t^*$'),fontsize=13)
		if add_title=='standard':
			plt.title('Hourly utility component',fontweight='bold',fontsize=16)
		elif add_title:
			plt.title(add_title,fontsize=13)

	#################################################################################################
	#									 	SAMPLING METHODS:										#
	#################################################################################################

	def draw_sample(self):
		if self.sample=='ph_gh':
			self.ph,self.gh,self.epsh,self.gh_std,self.eps_std = draw_gh_ph(self.n,self.loc,self.variance_p,self.variance_g,self.corr,self.seed)
		elif self.sample=='ph':
			self.ph,self.epsh,self.gh_std,self.eps_std = draw_ph(self.n,self.loc,self.variance_p,self.gh,self.corr,self.seed)
		elif self.sample=='gh':
			self.gh = draw_gh(self.n,self.variance_g,self.seed)
		self.p = sum(self.ph*self.gh)
		self.pmin = min(self.ph)

	def perturbed_samples(self,vec_variance):
		try:
			self.samples
		except AttributeError:
			self.samples = {}
		self.samples['variance_p']	= vec_variance
		self.samples['ph']			= np.empty([self.n,len(vec_variance)])
		for i in range(len(vec_variance)):
			self.samples['ph'][:,i] = draw_gh_ph(self.n,self.loc,vec_variance[i],self.variance_g,self.corr,self.seed)[0]
			self.samples['ph'][:,i] = self.samples['ph'][:,i]-(sum(self.samples['ph'][:,i]*self.gh)-self.p)
