#Simple SIR model

#RandomGraphs
#er=nx.erdos_renyi_graph(100,0.15)
#ws=nx.watts_strogatz_graph(30,3,0.1)
#ba=nx.barabasi_albert_graph(100,5)
#red=nx.random_lobster(100,0.9,0.9)
# for colour could use spreading as well as representing those that are infected susceptible
#http://networkx.github.io/documentation/latest/examples/drawing/random_geometric_graph.html
#TODO:
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
import matplotlib.pyplot as pl
import sys
import random

class Disease(object):
	"""Disease class used to manager recovery time, infection time"""
	def __init__(self,name,transmission_rate=0.65, recovery_time=3,r_prob=0.5,):
		self.name = name
		self.transmission_rate = transmission_rate
		self.recovery_time = recovery_time
		self.r_prob = r_prob

class Agent(object):
	"""Simple agent class for model infected susceptible recovered"""
	def __init__(self,n,t_prob, r_prob, infected=False,recovered=False,susceptible=True):
		self.infected = infected
		self.recovered = recovered
		self.susceptible = susceptible
		#Recovery rate
		self.r_prob = r_prob
		#Infection rate 
		self.t_prob = t_prob
		#keep track of time since disease
		self.dtime = 0
		self.n = n
	def infectAgent(self):
		if self.recovered == False and self.susceptible == True and self.infected == False:
			self.infected = True
			self.susceptible = False
			return True
		return False
			
	def immuneAgent(self):
		if self.infected == True and random.random() < self.r_prob:
			self.recovered = True
			self.infected = False
			self.susceptible = False
			return True
		return False

	def __repr__(self):
		return "%s" % self.n

class SIRmodel(object):
	"""Creates a simple SIR model object"""
	def __init__(self, modelname, disease, days=50, susceptible=0.99, infectious=0.01, population=200, d_prob=0.3, death_prob_dis=None, death_prob_norm=None, birth_rate=None ):
		self.name = modelname + disease.name
		#Set initial variables for model
		self.death_prob_dis = death_prob_dis
		self.death_prob_norm = death_prob_norm
		self.birth_rate = birth_rate
		self.infectious = infectious
		self.population = population
		self.r_prob = disease.r_prob
		self.susceptible = susceptible
		self.transmission_rate = disease.transmission_rate
		self.recovery_time = disease.recovery_time
		self.gamma =  1.0 / float(self.recovery_time) * self.r_prob
		#probability that in a random graph two nodes will be connected
		self.d_prob = d_prob
		self.agents = [ Agent(x,self.transmission_rate,self.r_prob) for x in xrange(self.population) ]
		self.days= days
		self.finished = 0
		self.inc = 1.0
		#Keep track of numbers in each compartment
		self.susceptibles = [0 for x in range(self.days)]
		self.infected = [0 for x in range(self.days) ]
		self.recoverers = [0 for x in range(self.days)]
		self.diseasenetwork = nx.Graph()
		self.output = self.run_sim(days)

	def randomise_recovery_agents(self):
		for agent in self.agents:
			recov_prob = random.random()
			agent.r_prob = recov_prob


	def run_sim(self, length,start=1, inc=1,g='nws',k=4):
		"""Run our simulation.
		Default will run for a 50 days one day at a time"""
		if(self.population < 0 ):
			sys.exit(1)
		self.diseasenetwork.add_nodes_from(self.agents)
		#Rewiring of graphs could count as random_mixing 
		#Types of random graphs
		gtype = { 'er':nx.fast_gnp_random_graph(self.population,self.d_prob),
				'nws':nx.newman_watts_strogatz_graph(self.population,3,self.d_prob),
				'cws':nx.connected_watts_strogatz_graph(self.population,2,self.d_prob),
				'ba':nx.barabasi_albert_graph(self.population,1),
				'grid':nx.grid_graph(dim=[self.population/2,self.population/2])}
		#This is causing the trouble need to map each edge to nodes :) 
		self.diseasenetwork.add_edges_from([ (self.agents[x],self.agents[y]) for x,y in gtype[g].edges() ])
		#keep track of compartments
		sList = [ x for x in self.agents ]
		iList = []
		rList = []
		for susAgent in sList:
			if random.random() < self.infectious:
				if susAgent.infectAgent():
					iList.append(susAgent)
					sList.remove(susAgent)

		#TODO implement graph for keeping track of infection
		infecGraph = nx.Graph()
		infecGraph.add_nodes_from(iList)
		initialNodes = iList
		#Should probably randomise infection but just testing
		#Randomly effect 5% of the population maybe
		print "Infected/Suscept count to start with : {} : {}  ".format(len(iList),len(sList))
		while(start <= length and len(iList) != 0):
			for agent in iList:
				for neighbour in self.diseasenetwork.neighbors(agent):
					if neighbour.susceptible and neighbour.t_prob >= random.random() and neighbour not in iList:
						infecGraph.add_edge(agent,neighbour)
						iList.append(neighbour)
						sList.remove(neighbour)
				#Time to recover probability is probably ok 
				#if agent.dtime > self.recovery_time:
				if agent.immuneAgent():
					rList.append(agent)
					iList.remove(agent)
				agent.dtime +=1

			if start % 10== 0 :

				pl.clf()
				pl.figure(figsize=(15,15),dpi=80)
				susp = [ node for node in self.diseasenetwork.nodes() if node.susceptible ]
				infec = [ node for node in self.diseasenetwork.nodes() if node.infected and not node.recovered ] 
				rec = [ node for node in self.diseasenetwork.nodes() if node.recovered ] 
				#position = nx.circular_layout(self.diseasenetwork)
				#position = nx.spring_layout(self.diseasenetwork,iterations=100)
				position = nx.graphviz_layout(self.diseasenetwork,prog='twopi',args='')
				nx.draw_networkx_nodes(self.diseasenetwork, position, nodelist=susp, node_color="b")
				nx.draw_networkx_nodes(self.diseasenetwork, position, nodelist=infec, node_color="r")
				nx.draw_networkx_nodes(self.diseasenetwork, position, nodelist=rec, node_color="g")

				nx.draw_networkx_edges(self.diseasenetwork,position)
				nx.draw_networkx_labels(self.diseasenetwork,position)
				cut = 1.00
   				xmax = cut * max(xx for xx, yy in position.values())
   				ymax = cut * max(yy for xx, yy in position.values())
    				pl.xlim(0, xmax)
   				pl.ylim(0, ymax)
				nx.draw(self.diseasenetwork)
				pl.savefig("{0}_SIRmodel_network_{1}.pdf".format(self.name,start))
				pl.clf()
				nx.draw(infecGraph,pos=nx.graphviz_layout(infecGraph,prog='twopi',args=''))
				pl.savefig("{0}_SIRmodel_network_{1}_infection_tree.pdf".format(self.name,start))
				pl.close()

			for neighbour in iList:
				neighbour.infectAgent()
			self.susceptibles[start-1] = (len(sList)/float(self.population))
			self.infected[start-1] = (len(iList)/float(self.population))
			self.recoverers[start-1] = (len(rList)/float(self.population))
			print start, len(iList), len(rList) , len(sList)

			start+=inc
		#We can change how we add random stuff later possible part of values
		#
		self.finished = start
		


	def plot_sim(self):
		"""Plots our model"""
		pl.clf()
		pl.figure(figsize=(8,8))
		print "Attempting to print model",self.susceptibles,self.infected
		pl.subplot(211)
		pl.plot(range(self.days), self.susceptibles, '.g', label='Susceptibles')
		pl.plot(range(self.days), self.recoverers, '.k', label='Recovered')
		pl.plot(range(self.days), self.infected, '.r', label='Infectious')
		pl.legend(loc=0)
		pl.title('Model for ' + self.name) 
		pl.xlabel('Time')
		pl.ylabel('% in compartments')
		#pl.subplot(212)
		#pl.xlabel('Time')
		#pl.ylabel('Infectious')
		pl.savefig(self.name + "plot.pdf")
	
	def return_basic_reproduction_num(self):
		"""Basic reproduction number is the average number of individuals that will be infected
		by an individual with a disease"""
		return self.r_prob / self.gamma * self.population

	def plot_determ(self,births=False):
		"""Deterministic plot taken from Modeling infectious diseases"""
		import scipy.integrate as spi
		import numpy as np
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
		pl.savefig(self.name + "determ_model_plot.pdf")
		pl.clf()
		pl.close()


def __main__():

	#for graphs in [ 'er','nws','cws','ba','grid']:
	#	for disease in [ Disease("slowrecov",recovery_time=6), Disease("Badrecov",r_prob=0.1),
	#			Disease("poortrans", transmission_rate=0.1), Disease("super",transmission_rate=0.9) ]:
	#		model = SIRmodel("Test"+graphs,disease, population=300)
	#		model.plot_sim()
	#		model.plot_determ()

	aids = Disease("aids",transmission_rate=0.4)
	aidsmod = SIRmodel("Test_increased_infecpop",aids,population=200)
	aidsmod.randomise_recovery_agents()
	aidsmod.plot_sim()
	aidsmod.plot_determ()

if __name__ == '__main__':
	__main__()













