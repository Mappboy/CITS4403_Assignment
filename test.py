from ComplexNetworkSim import NetworkAgent, Sim

class SIRSimple(NetworkAgent):
    """ an implementation of an agent following the simple SIR model """

    def __init__(self, state, initialiser):
        NetworkAgent.__init__(self, state, initialiser)
        self.infection_probability = 0.05 # 5% chance
        self.infection_end = 5

    def Run(self):
        while True:
            if self.state == SUSCEPTIBLE:
                self.maybeBecomeInfected()
                yield Sim.hold, self, NetworkAgent.TIMESTEP_DEFAULT #wait a step
            elif self.state == INFECTED:
                yield Sim.hold, self, self.infection_end  #wait end of infection
                self.state = RECOVERED
                yield Sim.passivate, self #remove agent from event queue

    def maybeBecomeInfected(self):
        infected_neighbours = self.getNeighbouringAgentsIter(state=INFECTED)
        for neighbour in infected_neighbours:
            if SIRSimple.r.random() < self.infection_probability:
                self.state = INFECTED
                break
