#Simple SIR model
# Three compartments 
# S = number susceptible
# I = number infectious
# R = number recovered 

import scipy.integrate as spi
import numpy as np

class SIRmodel(object):
	"""Creates a simple SIR model object"""
	def __init__(self, modelname, transmission_rate, recovery_rate, susceptible=0.9, infectious=1-1e-3, population=1000,death_prob_dis=None, death_prob_norm=None, birth_rate=None ):
		self.name = modelname
		self.death_prob_dis = death_prob_dis
		self.death_prob_norm = death_prob_norm
		self.birth_rate = birth_rate
		self.tranmission_rate = transmission_rate
		self.recovery_rate = recovery_rate
		self.susceptible = susceptible
		self.infectious = infectious
		self.population = population
		self.output = self.run_sim()

	def diff_eqs(self, input, time):
		"""Creates are model values """
		model = np.zeros((3))
		susceptible, infectious, time = input
		model[0] = -self.beta * susceptible * infectious
		model[1] = self.beta * susceptible * infectious  - self.recovery_rate* infectious
		model[2] = self.recovery_rate * infectious
		return model


	def run_sim(self, start=1, length=100, inc=1):
		"""Run our simulation.
		Default will run for a 100 days one day at a time"""
		sim_range = np.arange(start,length+inc, inc)
		return spi.odeint(self.diff_eqs,
				(self.susceptible,self.infectious,0),
				sim_range)

	def plot_sim(self):
		"""Plots our model"""
		from matplotlib import pyplot as pl
		pl.subplot(211)
		pl.plot(self.output[:,0], '.g', label='Susceptibles')
		pl.plot(self.output[:,2], '.k', label='Recovereds')
		pl.legend(loc=0)
		pl.title('Model for ' + self.model) 
		pl.xlabel('Time')
		pl.ylabel('Susceptibles and Recovereds')
		pl.subplot(212)
		pl.plot(self.output[:,1], '.r', label='Infectious')
		pl.xlabel('Time')
		pl.ylabel('Infectious')
		pl.show()

def __main__():
	aidsmod = SIRmodel("Test one", 2 , 0.02)
	SARSmod = SIRmodel("Test two", 4 , 0.5)
	aidsmod.plot_sim
	SARSmod.plot_sim

if __name__ == '__main__':
	__main__()













