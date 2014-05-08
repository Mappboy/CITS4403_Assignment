import random
import scipy
import pylab as pl

pl.figure()
for i in range(30):

	S = 99999
	I = 1
	R = 0

	N = S + I + R

	t = 0

	dt  = 1.0 / 24.0

	b = .01 * dt
	g = .01 * dt
	a = 0 * dt

	times = []
	iVals = []

	while I  > 0 and t <= 10000:
		leavingS = 0
		leavingI = 0
		
		if S > 0:
			leavingS = scipy.random.binomial(S, (((b*I) / N) + a))
		
		leavingI = scipy.random.binomial(I, g)
	
		S-= leavingS
		I+= leavingS
		I-= leavingI
		S+= leavingI
	
		iVals.append(I)
		times.append(t)
		t += dt

		#print ('t=',t,' S =', S, 'I =', I, 'R =', R)

	
	pl.plot(times,iVals)
pl.show()