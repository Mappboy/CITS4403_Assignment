#Simple SIR model
# Three compartments 
# S = number susceptible
# I = number infectious
# R = number recovered 


#nodes = [0,1,2,3]
#edges = [(0,1), (1,2), (3,1), (2,3)]
#nodeListA = [0,1]
#nodeListB = [2,3]    
#
#G = nx.Graph()
#G.add_nodes_from(nodes)
#G.add_edges_from(edges)
#position = nx.circular_layout(G)
#
#nx.draw_networkx_nodes(G,position, nodelist=nodeListA, node_color="b")
#nx.draw_networkx_nodes(G,position, nodelist=nodeListB, node_color="r")
#
#nx.draw_networkx_edges(G,position)
#nx.draw_networkx_labels(G,position)
#
#plt.show()

#RandomGraphs
#er=nx.erdos_renyi_graph(100,0.15)
#ws=nx.watts_strogatz_graph(30,3,0.1)
#ba=nx.barabasi_albert_graph(100,5)
#red=nx.random_lobster(100,0.9,0.9)
# for colour could use spreading as well as representing those that are infected susceptible
#http://networkx.github.io/documentation/latest/examples/drawing/random_geometric_graph.html
import networkx as nx
import matplotlib.pyplot as pl
import sys
import random



class Agent(object):
	"""Simple agent class for model infected susceptible recovered"""
	def __init__(self,n,t_prob, r_prob, infected=False,recovered=False,susceptible=True):
		self.infected = infected
		self.recovered = recovered
		self.susceptible = susceptible
		self.r_prob = r_prob
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
	def __init__(self, modelname, transmission_rate=0.65, recovery_time=3, days=50, r_prob=0.5, susceptible=0.9, infectious=1-1e-3, population=100, d_prob=0.6,death_prob_dis=None, death_prob_norm=None, birth_rate=None ):
		self.name = modelname
		self.death_prob_dis = death_prob_dis
		self.death_prob_norm = death_prob_norm
		self.birth_rate = birth_rate
		self.transmission_rate = transmission_rate
		self.recovery_time = recovery_time
		self.susceptible = susceptible
		self.infectious = infectious
		self.population = population
		self.r_prob = r_prob
		#probability that in a random graph two nodes will be connected
		self.d_prob = d_prob
		self.agents = [ Agent(x,self.transmission_rate,self.r_prob) for x in xrange(self.population) ]
		self.susceptibles = []
		self.infected = []
		self.days= days
		self.recoverers = []
		self.diseasenetwork = nx.Graph()
		self.output = self.run_sim(days)

	def run_sim(self, length,start=1, inc=1,randtype='cws'):
		"""Run our simulation.
		Default will run for a 100 days one day at a time"""
		if(self.population < 0 ):
			sys.exit(1)
		self.diseasenetwork.add_nodes_from(self.agents)
		#Types of random graphs
		randrun = { 'er':nx.fast_gnp_random_graph(self.population,self.d_prob),
				'kc':nx.karate_club_graph(),
				'nws':nx.newman_watts_strogatz_graph(self.population,3,self.d_prob),
				'cws':nx.connected_watts_strogatz_graph(self.population,2,self.d_prob),
				'ba':nx.barabasi_albert_graph(self.population,1)}
		#This is causing the trouble need to map each edge to nodes :) 
		self.diseasenetwork.add_edges_from([ (self.agents[x],self.agents[y]) for x,y in randrun[randtype].edges()])
		#keep track of compartments
		sList = [ x for x in self.agents ]
		iList = []
		rList = []
		for susAgent in sList:
			if random.random() < 0.25:
				if susAgent.infectAgent():
					iList.append(susAgent)
					sList.remove(susAgent)

		#TODO implement graph for keeping track of infection
		infecGraph = nx.Graph(iList)
		#Should probably randomise infection but just testing
		#Randomly effect 5% of the population maybe
		print "Infected/Suscept count to start with : {} : {}  ".format(len(iList),len(sList))
		while(start <= length):
			for agent in iList:
				for neighbour in self.diseasenetwork.neighbors(agent):
					if neighbour.susceptible and neighbour.t_prob < random.random() and neighbour not in iList:
						iList.append(neighbour)
						sList.remove(neighbour)
				if agent.dtime > self.recovery_time:
					if agent.immuneAgent():
						rList.append(agent)
						iList.remove(agent)
				agent.dtime +=1

			if start % 10== 0 :
				susp = [ node for node in self.diseasenetwork.nodes() if node.susceptible ]
				infec = [ node for node in self.diseasenetwork.nodes() if node.infected and not node.recovered ] 
				rec = [ node for node in self.diseasenetwork.nodes() if node.recovered ] 
				position = nx.circular_layout(self.diseasenetwork)
				nx.draw_networkx_nodes(self.diseasenetwork, position, nodelist=susp, node_color="b")
				nx.draw_networkx_nodes(self.diseasenetwork, position, nodelist=infec, node_color="r")
				nx.draw_networkx_nodes(self.diseasenetwork, position, nodelist=rec, node_color="g")

				nx.draw_networkx_edges(self.diseasenetwork,position)
				nx.draw_networkx_labels(self.diseasenetwork,position)

				nx.draw(self.diseasenetwork)
				pl.savefig("{0}_SIRmodel_network_{1}.pdf".format(self.name,start))

			for neighbour in iList:
				neighbour.infectAgent()
			self.susceptibles.append(len(sList)/float(self.population))
			self.infected.append(len(iList)/float(self.population))
			self.recoverers.append(len(rList)/float(self.population))
			print start, len(iList), len(rList) , len(sList)

			start+=inc
		#We can change how we add random stuff later possible part of values
		#
		


	def plot_sim(self):
		"""Plots our model"""
		pl.clf()
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

def __main__():
	aidsmod = SIRmodel("Test_increased_infecpop",population=500)
	aidsmod.plot_sim()

if __name__ == '__main__':
	__main__()













