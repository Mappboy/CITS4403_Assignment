#Simple SIR model

#RandomGraphs
#er=nx.erdos_renyi_graph(100,0.15)
#ws=nx.watts_strogatz_graph(30,3,0.1)
#ba=nx.barabasi_albert_graph(100,5)
#red=nx.random_lobster(100,0.9,0.9)
# for colour could use spreading as well as representing those that are infected susceptible
#http://networkx.github.io/documentation/latest/examples/drawing/random_geometric_graph.html
#TODO:
# Directed graph ?? 
#percolation rate
# rewire random graphs at time steps
# compare grahs using r ? or something 
# Make sure graphs are appropriately layed out
# Add in SIS model
# Add in SEIR
# Add in MSEIRS
# Add in deaths, births at each stage. 
# Add in weighted edges for familiarity
# Talk about GEMF
# Effect of different values in the agents and model infection rate

import networkx as nx
import numpy as np
import pylab
import matplotlib.pyplot as pl
import sys
import random

SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

class Disease(object):
	"""Disease class used to manager recovery time, infection time"""
	def __init__(self,name,transmission_rate=0.65, recovery_time=3,r_prob=0.5,):
		self.name = name
		self.transmission_rate = transmission_rate
		self.recovery_time = recovery_time
		self.r_prob = r_prob
	def __repr__(self):
		return "{} disease with beta = {} and gamma = {}".format(self.name,self.transmission_rate,self.r_prob)
	def __str__(self):
		return self.__repr__()

class Agent(object):
	"""Simple agent class for model infected susceptible recovered"""
	def __init__(self, n , t_prob, r_prob, state=SUSCEPTIBLE):
		self.state = SUSCEPTIBLE
		#Recovery rate
		self.r_prob = r_prob
		#Infection rate 
		self.t_prob = t_prob
		#keep track of time since disease
		self.dtime = 0
		self.n = n
	def infectAgent(self):
		if self.state == SUSCEPTIBLE:
			self.state = INFECTED
			return True
		return False
			
	def immuneAgent(self):
		if self.state == INFECTED and random.random() < self.r_prob:
			self.state = RECOVERED
			return True
		return False

	def __repr__(self):
		return "%s" % self.n

class SIRmodel(object):
	"""Creates a simple SIR model object"""
	def __init__(self, modelname, disease, modeltype='SIR', days=50, 
			infectious=0.1, population=400, d_prob=0.3, graphtype='reg',
			death_prob_dis=None, death_prob_norm=None, birth_rate=None ):
		self.name = modelname + disease.name
		self.disease = disease
		#Set initial variables for model
		self.death_prob_dis = death_prob_dis
		self.death_prob_norm = death_prob_norm
		self.birth_rate = birth_rate
		self.infectious = infectious
		self.population = population
		self.r_prob = disease.r_prob
		self.transmission_rate = disease.transmission_rate
		self.recovery_time = disease.recovery_time
		self.gamma =  1.0 / float(self.recovery_time) * self.r_prob
		#probability that in a random graph two nodes will be connected
		self.d_prob = d_prob
		self.graphtype = graphtype
		self.agents = [ Agent(x,self.transmission_rate,self.r_prob) for x in xrange(self.population) ]
		self.days= days
		self.finished = 0
		self.avgfinished = 0
		self.inc = 1.0
		#Keep track of numbers in each compartment
		self.susceptibles =  0
		self.infected = 0
		self.exposed = 0
		self.recovered = 0
		#Simulation and graph variables
		self.diseasenetwork = nx.Graph()
		self.isimulations = []
		self.rsimulations = []
		self.ssimulations = []
		self.infecGraph, self.output = self.run_sim(days,graphtype)

	def randomise_recovery_agents(self):
		for agent in self.agents:
			recov_prob = random.random()
			agent.r_prob = recov_prob

	def init_graph(self,g='reg',k=2):
		"""Creates a graph of type g"""
		self.diseasenetwork.add_nodes_from(self.agents)
		#Rewiring of graphs could count as random_mixing 
		#Types of random graphs
		gtype = { 'er':nx.fast_gnp_random_graph(self.population,0.05),
				'nws':nx.newman_watts_strogatz_graph(self.population,k,0.5),
				'ws':nx.watts_strogatz_graph(self.population,k,0.5),
				'cws':nx.connected_watts_strogatz_graph(self.population,k,0.5,10),
				'ba':nx.barabasi_albert_graph(self.population,k),
				'reg':nx.random_regular_graph(k,self.population),
				'grid':nx.grid_2d_graph(self.population/2,self.population/2) }
		#This is causing the trouble need to map each edge to nodes :) 
		if g == 'grid':
			self.diseasenetwork.add_edges_from([ (self.agents[x[0]],self.agents[y[0]]) for x,y in gtype[g].edges() ])
		else:
			self.diseasenetwork.add_edges_from([ (self.agents[x],self.agents[y]) for x,y in gtype[g].edges() ])



	def run_sim(self, length, graphtype, start=1, inc=1):
		"""Run our simulation.
		Default will run for a 50 days one day at a time"""
		if(self.population < 0 ):
			sys.exit(1)
		
		#First graph initialise graph
		if (len(self.isimulations) == 0):
			if graphtype:
				self.init_graph(graphtype)
			self.init_graph()
		
		iList = []
		#keep track of compartments
		for susAgent in self.agents:
			if random.random() < self.infectious:
				if susAgent.infectAgent():
					iList.append(susAgent)
					self.infected += 1
		self.susceptibles = self.population - self.infected
		count = [[],[],[]]
		#TODO implement graph for keeping track of infection
		infecTrace = nx.Graph()
		infecTrace.add_nodes_from(iList)
		#initialNodes = iList
		#Should probably randomise infection but just testing
		#Randomly effect 5% of the population maybe
		print "Infected/Suscept count to start with : {} : {} Graph is {} ".format(self.infected, self.susceptibles, graphtype)
		while(self.infected > 0 and start < self.days ):
			for agent in [ x for x in self.agents if x.state == INFECTED  ]:
				for neighbour in [ x for x in self.diseasenetwork.neighbors(agent) if x.state == SUSCEPTIBLE and x.t_prob <= random.random() ]:
					infecTrace.add_edge(agent,neighbour)
					iList.append(neighbour)
					neighbour.infectAgent()
					self.infected += 1
					self.susceptibles -= 1

				#Time to recover probability is probably ok 
				#if agent.dtime > self.recovery_time:
				if agent.immuneAgent():
					self.recovered +=1
					self.infected -=1
					iList.remove(agent)
				agent.dtime +=1

			#if start % 10== 0 :
				#fig = pylab.figure(figsize=(10,10),dpi=80)
				#self.plot_graphs(fig,start)

			count[SUSCEPTIBLE].append(self.susceptibles/float(self.population))
			count[INFECTED].append(self.infected/float(self.population))
			count[RECOVERED].append(self.recovered/float(self.population))
			print start, self.susceptibles, self.infected, self.recovered

			start+=inc
		#We can change how we add random stuff later possible part of values
		#
		self.finished = start
		self.avgfinished += start
		#Increment sim numbers
		self.isimulations.append(count[INFECTED])
		self.rsimulations.append(count[RECOVERED])
		self.ssimulations.append(count[SUSCEPTIBLE])
		return (infecTrace, count)

	def plot_inf_graph(self, save=False):
		pl.clf()
		nx.draw(self.infecGraph,pos=nx.springlayout(self.infecGraph))
		if save:
			pl.savefig("../graphs/{0}_SIRmodel_network_{1}_infection_tree.pdf".format(self.name,self.finished))
		else:
			pl.show()
		#nx.write_dot(infecGraph,"{0}_infection_tree_{1}.dot".format(self.name,timep))


	def plot_graphs(self,fig, timep,save=False):
		pylab.ion()
		susp = [ node for node in self.diseasenetwork.nodes() if node.state == SUSCEPTIBLE ]
		infec = [ node for node in self.diseasenetwork.nodes() if node.state == INFECTED ] 
		rec = [ node for node in self.diseasenetwork.nodes() if node.state == RECOVERED ] 
		#position = nx.circular_layout(self.diseasenetwork)
		position = nx.spring_layout(self.diseasenetwork,iterations=100)
		#position = nx.graphviz_layout(self.diseasenetwork,prog='twopi',args='')
		nx.draw_networkx_nodes(self.diseasenetwork, position, nodelist=susp, node_color="b")
		nx.draw_networkx_nodes(self.diseasenetwork, position, nodelist=infec, node_color="r")
		nx.draw_networkx_nodes(self.diseasenetwork, position, nodelist=rec, node_color="g")

		nx.draw_networkx_edges(self.diseasenetwork,position)
		nx.draw_networkx_labels(self.diseasenetwork,position)
		#cut = 1.00
   		#xmax = cut * max(xx for xx, yy in position.values())
   		#ymax = cut * max(yy for xx, yy in position.values())
    		#pl.xlim(0, xmax)
   		#pl.ylim(0, ymax)
		pl.pause(2)
		#nx.write_dot(self.diseasenetwork,"{0}_SIRmodel_network_{1}.dot".format(self.name,timep))
		if save:
			pl.savefig("../graphs/{0}_SIRmodel_network_{1}.pdf".format(self.name,timep))
		else:
			nx.draw(self.diseasenetwork)

	def write_graphs(self,infec,timep):
		pass

	def run_sim_ntimes(self,n):
		"""Calls run_sim number of times and takes average"""
		pl.clf()
		for eachsims in xrange(n):
			self.clear_arrays()
			self.run_sim(self.days,self.graphtype)
		avgrun = self.avgfinished / n 
		pl.clf()
		print "{} has average run time of {} for {} simulations".format(self.disease,avgrun,n)
		print self.ssimulations
		average_infec_sims = np.array(self.isimulations).mean(axis=0)
		average_recov_sims = np.array(self.rsimulations).mean(axis=0)
		average_susp_sims = np.array(self.ssimulations).mean(axis=0)
		pl.plot(average_infec_sims,'-b')
		pl.plot(average_recov_sims,'-r')
		pl.plot(average_susp_sims,'-g')
		pl.show()
		pl.savefig("../graphs/{0}_averages_infection_tree.pdf".format(self.name))


	def clear_arrays(self):
		"""Reset arrays"""
		self.susceptibles = 0
		self.infected = 0
		self.recoverers = 0

	def plot_sim(self,save=False):
		"""Plots our model"""
		pl.clf()
		pl.figure(figsize=(8,8))
		#pl.subplot(211)
		pl.plot(self.output[SUSCEPTIBLE], '.g', label='Susceptibles')
		pl.plot(self.output[RECOVERED], '.k', label='Recovered')
		pl.plot(self.output[INFECTED], '.r', label='Infectious')
		pl.legend(loc=0)
		pl.title('Model for ' + self.name) 
		pl.xlabel('Time')
		pl.ylabel('% in compartments')
		if save:
			pl.savefig(self.name + "plot.pdf")
		else:
			pl.show()
		pl.close()
	
	def return_basic_reproduction_num(self):
		"""Basic reproduction number is the average number of individuals that will be infected
		by an individual with a disease"""
		return self.r_prob / self.gamma * self.population

	def plot_determ(self,births=False):
		"""Deterministic plot taken from Modeling infectious diseases"""
		import scipy.integrate as spi
		#Taken from 
		S0=self.susceptible
		I0=self.infectious
		beta=self.transmission_rate
		INPUT = (S0, I0, 0.0)
		
		def diff_eqs(INP,t):  
			'''The main set of equations'''
			Y=np.zeros((3))
			V = INP    
			Y[0] = - beta * V[0] * V[1]
			Y[1] = beta * V[0] * V[1] - self.gamma * V[1]
			Y[2] = self.gamma * V[1]
			return Y   # For odeint
		t_start = 0.0; t_end = self.days; t_inc = self.inc 
		t_range = np.arange(t_start, t_end+t_inc, t_inc)
		RES = spi.odeint(diff_eqs,INPUT,t_range)
		pl.subplot(212)
		pl.plot(RES[:,0], '.g', label='Susceptibles')
		pl.plot(RES[:,1], '.r', label='Infectious')
		pl.plot(RES[:,2], '.k', label='Recovereds')
		pl.ylabel('Susceptibles and Recovereds')
		pl.xlim(0,self.days)
		pl.ylim(0,1.0)
		pl.title('Deterministic model')
		pl.xlabel('Time')
		pl.ylabel('Infectious')
		#pl.savefig(self.name + "determ_model_plot.pdf")
		pl.show()
		pl.clf()
		pl.close()


def __main__():

	#Output Statistics
	#Clustering coefficient, degree 
	for graphs in [ 'er','nws','cws','ba','grid','reg']:
		for disease in [ Disease("slowrecov",recovery_time=6), Disease("Badrecov",r_prob=0.1),
				Disease("poortrans", transmission_rate=0.01), Disease("super",transmission_rate=0.5) ]:
			model = SIRmodel("Test"+graphs,disease, population=100,graphtype=graphs)
			model.plot_sim(save=True)
			#model.run_sim_ntimes(10)
			#model.init_graph(g=graphs)
			#model.plot_determ()
	
	#Could randomise disease transmission
	#disease =  Disease("", transmission_rate=0.001, r_prob=0.01, recovery_time=3)
	#print "Random Graph"
	#randmodel = SIRmodel("Test_random",disease, population=100,graphtype='ba')
	#randmodel.plot_sim()
	#randmodel.plot_inf_graph()
	#randmodel.run_sim_ntimes(10)
	#print "Small World"
	#smallworldmodel = SIRmodel("Test_SmallWorld",disease, population=100,graphtype='nws')
	#smallworldmodel.run_sim_ntimes(10)
	#print "Scale Free"
	#scalemodel = SIRmodel("Test_random",disease, population=100,graphtype='ba')
	#scalemodel.run_sim_ntimes(10)
	#print "Lattice"
	#gridmodel = SIRmodel("Test_random",disease, population=100,graphtype='grid')
	#gridmodel.run_sim_ntimes(10)

	#aids = Disease("aids",transmission_rate=0.05,r_prob=0.2)
	#aidsmod = SIRmodel("Test_increased_infecpop",aids,population=50)
	##aidsmod.randomise_recovery_agents()
	##aidsmod.run_sim_ntimes(10)
	#aidsmod.plot_sim()
	#aidsmod.plot_determ()

if __name__ == '__main__':
	__main__()













