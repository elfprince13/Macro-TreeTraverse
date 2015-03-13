#include "treedefs.h"
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>

template<typename T> T factorial(T n)
{
  return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

template<typename Float> Float readFloat(FILE *f) {
  Float v;
  fread((void*)(&v), sizeof(v), 1, f);
  return v;
}


template<size_t DIM, typename Float> Float sys_energy(size_t n, Particle<DIM, Float> const *particles){
	Float energy = 0.0;
	for(size_t i = 0; i < n; i++){
		energy += 0.5 * particles[i].m * mag_sq(particles[i].vel);
		for(size_t j = 0; j < i; j++){
			energy -= particles[i].m * particles[j].m / mag(particles[i].pos - particles[j].pos);
		}
	}
	return energy;
}

template<size_t ORDER, typename Float> Float forward_euler(Vec<ORDER+1, Float> yp, Float dt){
	Float yn = 0.0;
	for(size_t i = 0; i <= ORDER; i++){
		yn += (yp.x[i] / factorial(i)) * pow(dt, i);
	}
	return yn;
}

template<size_t DIM, typename Float> void integrate_system(size_t n, Particle<DIM, Float> *particles, Vec<DIM, Float> const *forces, Float dt){
	for(size_t i = 0; i < n; i++){
		for(size_t j = 0; j < DIM; j++){
			Vec<3, Float> posp = {.x = {particles[i].pos.x[j], particles[i].vel.x[j], forces[i].x[j] / particles[i].m}};
			Vec<2, Float> velp = {.x = {particles[i].vel.x[j], forces[i].x[j] / particles[i].m}};
			particles[i].pos.x[j] = forward_euler<2, Float>(posp, dt);
			particles[i].vel.x[j] = forward_euler<1, Float>(velp, dt);	
		}
	}
	
}

template<size_t DIM, typename Float> void calc_forces_bruteforce(size_t n, Particle<DIM, Float> const *particles, Vec<DIM, Float> *forces){
	for(size_t i = 0; i < n; i++){
		forces[i] = (Vec<DIM,Float>){.x = {0, 0, 0}};
		for(size_t j = 0; j< n; j++){
			if(i == j) j++;
			Vec<DIM, Float> disp = particles[i].pos - particles[j].pos;
			forces[i] = (forces[i] + disp * ((particles[i].m * particles[j].m) / (Float)pow(mag_sq(disp),1.5)));
		}
	}
}

template<size_t DIM, typename Float> Vec<DIM, Float> min_extents(size_t n, Particle<DIM, Float> const *particles){
	Vec<DIM, Float> minE;
	for(size_t i = 0; i < DIM; i++){
		minE.x[i] = std::numeric_limits<Float>::infinity();
	}
	for(size_t i = 0; i < n; i++){
		minE = min(minE, particles[i].pos);
	}
	return minE;
}

template<size_t DIM, typename Float> Vec<DIM, Float> max_extents(size_t n, Particle<DIM, Float> const *particles){
	Vec<DIM, Float> maxE;
	for(size_t i = 0; i < DIM; i++){
		maxE.x[i] = -std::numeric_limits<Float>::infinity();
	}
	for(size_t i = 0; i < n; i++){
		maxE = max(maxE, particles[i].pos);
	}
	return maxE;
}

template<size_t DIM, size_t MAX_LEVELS, typename Float> void delete_tree(Node<DIM, Float>**tree){
	for(size_t i = 0; i < MAX_LEVELS; i++){
		delete [] tree[i];
	}
	delete [] tree;
}

// Assume particles are sorted by z-Order;
template<size_t DIM, size_t MAX_LEVELS, size_t NODE_THRESHOLD, typename Float>
void add_level(Node<DIM, Float> **levels, size_t level, Vec<DIM, Float> minExtents, Vec<DIM, Float> maxExtents,
				   std::vector< Particle<DIM, Float> > particleV, Particle<DIM, Float>  *particlesDst, size_t node_counts[MAX_LEVELS], size_t &pCount){
	Node<DIM, Float> nodeHere;
	nodeHere.minX = minExtents;
	nodeHere.maxX = maxExtents;
	//std::cout << "Inserting " << particleV.size() << " particles at " << level << std::endl;
	if((level+1 == MAX_LEVELS) || (particleV.size() < NODE_THRESHOLD)){
		nodeHere.isLeaf = true;
		nodeHere.childCount = particleV.size();
		nodeHere.childStart = pCount;
		Float mass = 0.0;
		Vec<DIM, Float> bary; for(size_t i = 0; i < DIM; i++){ bary.x[i] = 0.0; };
		for(auto it = particleV.begin(); it != particleV.end(); ++it){
			particlesDst[pCount++] = *it;
			mass += (*it).m;
			bary = bary + ((*it).pos * (*it).m);
		}
		bary = bary / mass;
	} else {
		nodeHere.isLeaf = false;
		nodeHere.childStart = std::numeric_limits<size_t>::max();
		nodeHere.childCount = 0;
		
		std::vector< Particle<DIM, Float> > partBuffer[1 << DIM];
		Vec<DIM, Float> minXs[1 << DIM];
		Vec<DIM, Float> maxXs[1 << DIM];
		Vec<DIM, Float> mid = minExtents + ((maxExtents - minExtents) / 2);
		
		
		
		for(size_t q = 0; q < (1 << DIM); q++){
			partBuffer[q].clear();
			for(size_t i = 0; i < DIM; i++){
				if(q & (1 << i)){
					minXs[q].x[i] = minExtents.x[i];
					maxXs[q].x[i] = mid.x[i];
				} else {
					minXs[q].x[i] = mid.x[i];
					maxXs[q].x[i] = maxExtents.x[i];
				}
			}
		}
		for(auto it = particleV.begin(); it != particleV.end(); ++it){
			size_t q;
			for(q = 0; q < (1 << DIM); q++){
				if(contains(minXs[q], maxXs[q], (*it).pos)){
					partBuffer[q].push_back(*it);
					break;
				}
			}
			if (q == (1 << DIM)) {
				std::cerr << "Couldn't place a particle" << std::endl;
				exit(1);
			}
		}
		for(size_t q = 0; q < (1 << DIM); q++){
			if(partBuffer[q].size() != 0){
				if(node_counts[MAX_LEVELS] < nodeHere.childStart){
					nodeHere.childStart = node_counts[MAX_LEVELS];
				}
				add_level<DIM, MAX_LEVELS, NODE_THRESHOLD, Float>(levels, level+1, minXs[q], maxXs[q], partBuffer[q], particlesDst, node_counts, pCount);
				nodeHere.childCount++;
			}
		}
	}
	levels[level][node_counts[level]++] = nodeHere;
}

template<size_t DIM, size_t MAX_LEVELS, size_t NODE_THRESHOLD, typename Float>
Node<DIM, Float>** build_tree(size_t n, const Particle<DIM, Float> *particles, Particle<DIM, Float> *particlesDst, size_t node_counts[MAX_LEVELS]) {
	Node<DIM, Float> **levels = new Node<DIM, Float>*[MAX_LEVELS];
	for(size_t i = 0; i < MAX_LEVELS; i++){
		levels[i] = new Node<DIM, Float>[1 << (DIM * i)];
		node_counts[i] = 0;
	}
	size_t pCount = 0;
	std::vector< Particle<DIM, Float> > particleV(particles, particles + n);
	add_level<DIM, MAX_LEVELS, NODE_THRESHOLD, Float>(levels, 0, min_extents<DIM, Float>(n, particles), max_extents<DIM, Float>(n, particles),  particleV, particlesDst, node_counts, pCount);
	if(pCount != n){
		std::cerr << "Count mismatch" << n << " != " << pCount << std::endl;
		exit(1);
	}
	return levels;
}

template<size_t DIM, size_t MAX_LEVELS, size_t N_GROUP, typename Float>
std::vector<GroupInfo<DIM, Float, N_GROUP> > groups_from_tree(Node<DIM, Float>* tree[MAX_LEVELS], size_t node_counts[MAX_LEVELS], const Particle<DIM, Float> *particles){
	std::vector<GroupInfo<DIM, Float, N_GROUP> > groups;
	groups.clear();
	for(size_t level = 0; level < MAX_LEVELS; level++){
		for(size_t node = 0; node < node_counts[level]; node++){
			// We probably don't want to load balance groups because under-full groups can ante-up per-particles
			for(size_t i = 0; i + N_GROUP < tree[level][node].childCount; i += N_GROUP){
				GroupInfo<DIM, Float, N_GROUP> group;
				group.childStart = tree[level][node].childStart + i;
				group.childCount = (i < tree[level][node].childCount) ? N_GROUP : (tree[level][node].childCount % N_GROUP);
				group.minX = min_extents(group.childCount, particles + group.childStart);
				group.maxX = max_extents(group.childCount, particles + group.childStart);
				
				groups.push_back(group);
			}
		}
	}
	return groups;
}




#define DIM 3
#define Float float
#define DT 0.001

#define MAX_LEVELS 8
#define NODE_THRESHOLD 16
#define N_GROUP 16

int main(int argc, char* argv[]) {
	int nPs = atoi(argv[1]);
	Particle<DIM,Float> *bodies = new Particle<DIM,Float>[nPs] ;
	Particle<DIM,Float> *bodiesSorted = new Particle<DIM,Float>[nPs] ;
	Vec<DIM,Float> *forces = new Vec<DIM,Float>[nPs];
	FILE *f = fopen(argv[2],"rb");
	for(size_t i = 0; i < nPs; i++){
		bodies[i].m = readFloat<Float>(f);
		for(size_t j = 0; j < DIM; j++){
			bodies[i].pos.x[j] = readFloat<Float>(f);		
		}
		for(int j = 0; j < DIM; j++){
			bodies[i].vel.x[j] = readFloat<Float>(f);
		}
	}
	fclose(f);
	
	Float e_init = sys_energy<DIM, Float>(nPs, bodies);
	
	std::cout << nPs << "\t" << (1.0  / nPs) << "\t" << bodies[0].m << "\t" << bodies[nPs - 1].m << std::endl;
	std::cout << "Init energy:\t" << e_init << std::endl;
	for(int i = 0; i < 10; i++){
		calc_forces_bruteforce<DIM, Float>(nPs, bodies, forces);
		integrate_system<DIM, Float>(nPs, bodies, forces, DT);
		Float e_now = sys_energy<DIM, Float>(nPs, bodies);
		std::cout << "%dE:\t" << (e_init - e_now) / e_init << std::endl;
	}
	
	/*
	std::vector< Particle<DIM, Float> > particleV(bodies, bodies + nPs);
	ParticleComparator<DIM, Float> comp;
	comp.minX = min_extents<DIM, Float>(nPs, bodies);
	comp.maxX = max_extents<DIM, Float>(nPs, bodies);
	std::sort(particleV.begin(), particleV.end(), comp);
	 */
	
	// This would be more efficient with proper vectors, but it's fine for now :)
	// At 8 levels deep, it won't be a problem
	size_t node_counts[MAX_LEVELS];
	size_t validateCt = 0;
	Node<DIM, Float>** tree = build_tree<DIM, MAX_LEVELS, NODE_THRESHOLD, Float>(nPs, bodies, bodiesSorted, node_counts);
	for(size_t level = 0; level < MAX_LEVELS; level++){
		for(size_t nodeI = 0; nodeI < node_counts[level]; nodeI++){
			//std::cout << "Node at level " << level << " has leaf " << tree[level][nodeI].isLeaf << " and " << tree[level][nodeI].childCount << " children" << std::endl;
			validateCt += tree[level][nodeI].isLeaf ? tree[level][nodeI].childCount : 0;
		}
	}
	
	std::cout << "Total in leaves:\t" << validateCt << "\tvs\t" << nPs << "\tto start "<< std::endl;
	
	std::vector<GroupInfo<DIM, Float, N_GROUP> > groups = groups_from_tree<DIM, MAX_LEVELS, N_GROUP>(tree, node_counts, bodiesSorted);
	
	delete_tree<DIM, MAX_LEVELS, Float>(tree);
	
	
	
	
	delete [] bodies;
	delete [] bodiesSorted;
}
