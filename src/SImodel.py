import random


from Graph import *
from RandomGraph import *
from SmallWorldGraph import *
from SmallWorldGraph import *
from matplotlib import pyplot as pl
import numpy as np

#TODO



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
		if infec <=  0 and days >= 100 or susps == npop:
			if not recover and susps <= 0:
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

def main(script, n=20,k=5, kval=0.1, beta=0.05, gamma=0.2, infected=3, infec_len=3, recover='False', graphtype='rand', *args):
    n = int(n)
    kval = float(kval)
    beta = float(beta)
    gamma = float(gamma)
    infected = int(infected)
    infec_len= int(infec_len)
    recover= bool(recover)
    k = int(k)
    #p = float(p)
    #num = int(num)
    #count = test_p(n, p, num)

    #show_graph(g)
    #Ten trials 
    #infecsim = []
    #labels = []
    graphs = {'rand':RandomGraph(create_nverts(n)),
		    'small':SmallWorldGraph(create_nverts(n),k,kval)}
    g= graphs[graphtype]
    if graphtype == 'rand':
	    g.add_random_edges(kval)
    runvals = run_epidemic(g,beta,gamma,infected,infection_length=infec_len,recover=recover)
    plot_compartments(runvals,recover)

#    for n in xrange(10):
#        g= RandomGraph(create_nverts(20))
#        g.add_random_edges(randk)
#	args = [g,0.05,0.2,3,0.05]
#	runvals = run_n_times(10,args)
#    	#runvals = run_epidemic(g,0.05,0.2,3,0.05)
#	infecsim.append(runvals)
#	labels.append(str(randk))
#	randk+=0.1
    #print infecsim
    #plot_infected(finishinfex(infecsim),labels)
    #plot_compartments(runvals)


    #show_graph(g)
    #print count

if __name__ == '__main__':
    import sys
    main(*sys.argv)

