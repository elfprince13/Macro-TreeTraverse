#!/usr/bin/env python
class Particle(object):
	def __init__(self, mass=0, pos=(0, 0, 0), vel=(0, 0, 0)):
		self.mass = mass
		self.pos = pos
		self.vel = vel

import math, random

def sph2xyz(radius, theta, phi):
	return tuple(radius * unit for unit in (math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi),math.cos(theta)))


def mkuniform(ln,ppn,seed=None):
	"""This is the algorithm from The Art of Computational Science, vol. 9, ch. 2"""
	if seed != None:
		random.seed(seed)
	n = 2**ln
	nb = [Particle() for i in range((n**3)*ppn)]
	scale = 1.0 / n
	for i,b in enumerate(nb):
		b.mass = 1.0/(n*ppn)
		pN = i / ppn
		x = pN % n
		y = (pN / n) % n
		z = (pN / (n*n))
		radius = (0.01*scale) / ( random.random() ** (-2.0/3.0) - 1.0) ** 0.5
		theta = math.acos(random.uniform(-1.0, 1.0))
		phi = random.uniform(0, math.pi * 2)
		b.pos = tuple(c+cs for (c,cs) in zip((x,y,z),sph2xyz(radius, theta, phi)))
		x = 0.0
		y = 0.1
		while y > (x**2 * (1 - x**2)**3.5):
			x = random.uniform(0, 1.0)
			y = random.uniform(0,0.1)
		velocity = x * (2.0 ** 0.5) * (1.0 + radius ** 2)**(-0.25)
		theta = math.acos(random.uniform(-1.0,1.0))
		phi = random.uniform(0, math.pi * 2)
		b.vel = sph2xyz(0.01*scale*velocity/ppn, theta, phi)
	return nb

if __name__ == "__main__":
	import struct,sys
	if len(sys.argv) in {2,4}:
		ln = int(sys.argv[1])
		ppn = int(sys.argv[2] if len(sys.argv) >= 3 else 1)
		s = long(sys.argv[2]) if len(sys.argv) == 4 else None 
		print "".join("".join(struct.pack('<f',f) for f in (p.mass,)+p.pos+p.vel) for p in mkuniform(ln,ppn,s))
	else:
		sys.exit(1)
