import string
import random

from collections import deque

from Graph import *
from matplotlib import pyplot as pl
import numpy as np

STATES = [SUSCEPTIBLE,INFECTED,INFECTED,RECOVERED]
class RandomGraph(Graph):
    """An Erdos-Renyi random graph is a Graph where the probability 
    of an edge between any two nodes is (p).
    """
    def __init__(self,nv=[],ev=[]):
	Graph.__init__(self,nv,ev)

    def add_random_edges(self, p=0.05):
        """Starting with an edgeless graph, add edges to
        form a random graph where (p) is the probability 
        that there is an edge between any pair of vertices.
        """
        vs = self.vertices()
        for i, v in enumerate(vs):
            for j, w in enumerate(vs):
                if j <= i: continue
                if random.random() > p: continue
                self.add_edge(Edge(v, w))

    def visit(self,v):
	v.visited = True

    def bfs(self, s, visit=None):
        """Breadth first search, starting with (s).
        If (visit) is provided, it is invoked on each vertex.
        Returns the set of visited vertices.
        """
        visited = set()

        # initialize the queue with the start vertex
        queue = deque([s])
        
        # loop until the queue is empty
        while queue:

            # get the next vertex
            v = queue.popleft()

            # skip it if it's already visited
            if v in visited: 
		    continue

            # mark it visited, then invoke the visit function
            visited.add(v)
            if visit: 
		    visit(v)

            # add its out vertices to the queue
            queue.extend(self.out_vertices(v))

        # return the visited vertices
        return visited

    def is_connected(self):
        """Returns True if there is a path from any vertex to
        any other vertex in this graph; False otherwise.
        """
        vs = self.vertices()
        visited = self.bfs(vs[0])
        return len(visited) == len(vs)


def show_graph(g):
    import GraphWorld

    for v in g.vertices():
        if v.state == SUSCEPTIBLE: 
            v.color = 'white'
	elif v.state == RECOVERED:
		v.color = 'blue'
        else:
            v.color = 'red'

    layout = GraphWorld.CircleLayout(g)
    gw = GraphWorld.GraphWorld()
    gw.show_graph(g, layout)
    gw.mainloop()


def test_graph(n, p):
    """Generates a random graph with (n) vertices and probability (p).
    Returns True if it is connected, False otherwise
    """
    labels = string.lowercase + string.uppercase + string.punctuation
    vs = [Vertex(c) for c in labels[:n]]
    g = RandomGraph(vs)
    g.add_random_edges(p=p)
    # show_graph(g)
    return g.is_connected()


def test_p(n, p, num):
    """Generates (num) random graphs with (n) vertices and
    probability (p) and return the count of how many are connected.
    """
    count = 0
    for i in range(num):
        if test_graph(n, p):
            count += 1
    return count

def run_epidemic(G,beta,gamma,ninfected,infection_length=2, pgrowth=0.0,growth=False, recover=False):
	if pgrowth > 1 or pgrowth < -1:
		return None
	population = G.vertices()
	npop = float(len(population))
	recov = 0
	for n in xrange(ninfected):
		patient_zero = random.choice(population)
		patient_zero.state = INFECTED
		patient_zero.infect_time = 0
	infec = ninfected
	susps = len(population) - infec
	counts = [[],[],[]]
	days = 0
	while ( True ):
		#Change population according to growth rate
		if growth:
			population = G.vertices()
			npop = float(len(population))
			#Get floor and add new nodes
			newpop = int(pgrowth*npop)
			susps+=newpop
			for n in xrange(abs(newpop)):
				if pgrowth > 0:
					newv = Vertex( str(npop+1) )
					G.add_vertex(newv)
					G.add_edge(Edge(newv,random.choice(population)))
				else:
					G.remove_vertex(random.choice(population))
	
		#Actual part running simulation
		days +=1
		for infected in [ x for x in population if x.state == INFECTED ]:
			for neighbours in G.out_vertices(infected):
				if random.random() < beta and neighbours.state == SUSCEPTIBLE:
					neighbours.state = INFECTED
					neighbours.infect_time = days
					infec+=1
					susps-=1

			#If adding people to recover
			if recover and random.random() < gamma:
				infected.state = RECOVERED
				infec -=1
				recov +=1
			elif not recover and ( infected.infect_time + infection_length == days) :
				infected.state = SUSCEPTIBLE
				infec -=1
				susps += 1
		counts[SUSCEPTIBLE].append(susps/npop)
		counts[INFECTED].append(infec/npop)
		#SIR model
		if recover:
			counts[RECOVERED].append(recov/npop)
		#Check if we should end simulation
		if infec <=  0 and days >= 100:
			if not recover and susp <= 0:
				break
			break
	print "Ran for {} days".format(days)
	return counts


def plot_compartments(compvals,recover=False):
    pl.ylabel('Susceptibles,Recovered,Infected')
    pl.ylim(0,1.0)
    pl.title('Epidemic model')
    pl.xlabel('Time')
    pl.plot(compvals[SUSCEPTIBLE],'-g',label='Susceptible')
    pl.plot(compvals[INFECTED],'-b',label='Infected')
    if recover:
    	pl.plot(compvals[RECOVERED],'-r',label='Recovered')
    pl.legend(loc=0) 
    pl.show()

def run_n_times(n, args):
	counts = []
	for sim in xrange(n):
		counts.append(run_epidemic(*args))
	return np.average(map(finishinfex,counts),axis=0)
	
def plot_infected(infections,labels):
    pl.clf()
    colormap = pl.cm.gist_ncar
    pl.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(infections))])
    
    averageline = np.average(infections,axis=0)
    pl.ylabel('Infections')
    pl.ylim(0,1.0)
    pl.title('Epidemic model')
    pl.xlabel('Time')
    for l,infec in enumerate(infections):
    	pl.plot(infec,label=labels[l])
    pl.plot(averageline,'-r',label='Average')
    labels.append('Average')
    pl.legend(labels, ncol =3,loc='upper left')
    pl.show()
	#TODO plot infections multiple


def create_nverts(n):
	alnumgen = alphnum(n)
	x = [ Vertex(alnumgen.next()) for a in xrange(n)  ]  
	return x[:n]

def alphnum(n):
        i = 1
        while True and i < n:
                for c in string.lowercase:
                        yield c + str(i)
                        if ( c == "z" ):
                                i+=1

def finishinfex(infecs,average=False):
    """Create a new array that based on maximum days in simulation. Required to get average"""
    n_sims = len(infecs)
    n_infectlen = max(map(len, infecs))

    new_array = np.empty((n_sims, n_infectlen))
    new_array.fill(0)
    for i, row in enumerate(infecs):
        for j, ele in enumerate(row):
            new_array[i, j] = ele
    if average:
	    pass
    return new_array

def main(script, n=20, p=0.1, num=1, infected=3, infec_len=3, *args):
    n = int(n)
    p = float(p)
    num = int(num)
    count = test_p(n, p, num)

    #show_graph(g)
    #Ten trials 
    randk = 0.1
    g= RandomGraph(create_nverts(n))
    g.add_random_edges(randk)


    #show_graph(g)
    test_graph(10,0.6)
    print count

dostuff = []
def visit(v):
    dostuff.append(v)

def testbfs(graph):
	"""Breadth-first search on a graph, starting at top_node."""
	top_node = graph.vertices()[0]
    	visited = set()
    	queue = [top_node]
    	while len(queue):
	    curr_node = queue.pop(0)    # Dequeue
	    visit(curr_node)            # Visit the node
	    visited.add(curr_node)

    	   # Enqueue non-visited and non-enqueued children
	    queue.extend(c for c in graph.out_vertices(curr_node)
    	                if c not in visited and c not in queue)

if __name__ == '__main__':
    import sys
    main(*sys.argv)
