#!/usr/bin/env python
class Particle(object):
	def __init__(self, mass=0, pos=(0, 0, 0), vel=(0, 0, 0)):
		self.mass = mass
		self.pos = pos
		self.vel = vel

import math, random

def sph2xyz(radius, theta, phi):
	return tuple(radius * unit for unit in (math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi),math.cos(theta)))


def mkplummer(n, seed=None):
	"""This is the algorithm from The Art of Computational Science, vol. 9, ch. 2"""
	if seed != None:
		random.seed(seed)
	nb = [Particle() for i in range(n)]
	for b in nb:
		b.mass = 1.0/n
		radius = 1.0 / ( random.random() ** (-2.0/3.0) - 1.0) ** 0.5
		theta = math.acos(random.uniform(-1.0, 1.0))
		phi = random.uniform(0, math.pi * 2)
		b.pos = sph2xyz(radius, theta, phi)
		x = 0.0
		y = 0.1
		while y > (x**2 * (1 - x**2)**3.5):
			x = random.uniform(0, 1.0)
			y = random.uniform(0,0.1)
		velocity = x * (2.0 ** 0.5) * (1.0 + radius ** 2)**(-0.25)
		theta = math.acos(random.uniform(-1.0,1.0))
		phi = random.uniform(0, math.pi * 2)
		b.vel = sph2xyz(velocity, theta, phi)
	return nb

if __name__ == "__main__":
	import struct,sys
	if len(sys.argv) in {2,3}:
		n = int(sys.argv[1])
		s = long(sys.argv[2]) if len(sys.argv) == 3 else None 
		print "".join("".join(struct.pack('<f',f) for f in (p.mass,)+p.pos+p.vel) for p in mkplummer(n,s))
	else:
		sys.exit(1)
