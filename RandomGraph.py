import string
import random

from collections import deque

from Graph import *


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

def run_epidemic(G,beta,gamma,ninfected):
	population = G.vertices()
	npop = float(len(population))
	recov = 0
	for n in xrange(ninfected):
		random.choice(population).state = INFECTED
	infec = ninfected
	susps = len(population) - infec
	counts = [[],[],[]]
	days = 0
	while ( infec != 0):
		if ( susps + infec + recov != len(population) ):
			break
		days +=1 
		for infected in [ x for x in population if x.state == INFECTED ]:
			for neighbours in G.out_vertices(infected):
				if random.random() < beta and neighbours.state == SUSCEPTIBLE:
					neighbours.state = INFECTED
					infec+=1
					susps-=1

			if random.random() < gamma:
				infected.state = RECOVERED
				infec -=1
				recov +=1
		counts[SUSCEPTIBLE].append(susps/npop)
		counts[INFECTED].append(infec/npop)
		counts[RECOVERED].append(recov/npop)
	print "Ran for {} days".format(days)
	return counts




def create_nverts(n):
	x = [ Vertex(a) for a in string.lowercase if a < chr(ord('a')+n)  ]  
	return x[:n]

def main(script, n=26, p=0.1, num=1, *args):
    from matplotlib import pyplot as pl
    n = int(n)
    p = float(p)
    num = int(num)
    count = test_p(n, p, num)
    g= RandomGraph(create_nverts(100))
    g.add_random_edges(0.1)
    show_graph(g)
    runvals = run_epidemic(g,0.05,0.2,3)
    pl.ylabel('Susceptibles,Recovered,Infected')
    pl.ylim(0,1.0)
    pl.title('Epidemic model')
    pl.xlabel('Time')
    pl.plot(runvals[SUSCEPTIBLE],'-g',label='Susceptible')
    pl.plot(runvals[INFECTED],'-b',label='Infected')
    pl.plot(runvals[RECOVERED],'-r',label='Recovered')
    pl.legend(loc=0) 
    pl.show()
    show_graph(g)
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
