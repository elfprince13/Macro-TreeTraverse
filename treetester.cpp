#include "treedefs.h"
#include "treecodeCU.h"
#include "TransposalTools.h"
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <type_traits>
#include "Miniball.hpp"

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
			Float posA[3] = {particles[i].pos.x[j], particles[i].vel.x[j], forces[i].x[j] / particles[i].m};
			Float velA[2] = {particles[i].vel.x[j], forces[i].x[j] / particles[i].m};
			Vec<3, Float> posp(posA);
			Vec<2, Float> velp(velA);
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
		nodeHere.radius = 0.0;
		Float mass = 0.0;
		Vec<DIM, Float> bary; for(size_t i = 0; i < DIM; i++){ bary.x[i] = 0.0; };
		for(auto it = particleV.begin(); it != particleV.end(); ++it){
			particlesDst[pCount++] = *it;
			mass += (*it).m;
			bary = bary + ((*it).pos * (*it).m);
		}
		bary = bary / mass;
		nodeHere.mass = mass;
		nodeHere.barycenter = bary;
		for(auto it = particleV.begin(); it != particleV.end(); ++it){
			Float radHere = mag((*it).pos - bary);
			nodeHere.radius = radHere > nodeHere.radius ? radHere : nodeHere.radius;
		}
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
				if(node_counts[level+1] < nodeHere.childStart){
					nodeHere.childStart = node_counts[level+1];
				}
				add_level<DIM, MAX_LEVELS, NODE_THRESHOLD, Float>(levels, level+1, minXs[q], maxXs[q], partBuffer[q], particlesDst, node_counts, pCount);
				nodeHere.childCount++;
			}
		}
	}
	levels[level][node_counts[level]++] = nodeHere;
}

template<size_t MAX_LEVELS> inline void clear_node_counts(size_t node_counts[MAX_LEVELS]){
	for(size_t i = 0; i < MAX_LEVELS; i++){
		node_counts[i] = 0;
	}
}

template<size_t DIM, size_t MAX_LEVELS, size_t NODE_THRESHOLD, typename Float>
void insert_all(size_t n, const Particle<DIM, Float> *particles, Particle<DIM, Float> *particlesDst,
							  Node<DIM, Float> **levels, size_t node_counts[MAX_LEVELS]) {
	size_t pCount = 0;
	std::vector< Particle<DIM, Float> > particleV(particles, particles + n);
	add_level<DIM, MAX_LEVELS, NODE_THRESHOLD, Float>(levels, 0, min_extents<DIM, Float>(n, particles), max_extents<DIM, Float>(n, particles),  particleV, particlesDst, node_counts, pCount);
	if(pCount != n){
		std::cerr << "Count mismatch" << n << " != " << pCount << std::endl;
		exit(1);
	}
}

template<size_t DIM, size_t MAX_LEVELS, size_t NODE_THRESHOLD, typename Float>
Node<DIM, Float>** build_tree(size_t n, const Particle<DIM, Float> *particles, Particle<DIM, Float> *particlesDst,
							  size_t node_counts[MAX_LEVELS]) {
	
	Node<DIM, Float> **levels = new Node<DIM, Float>*[MAX_LEVELS];
	for(size_t i = 0; i < MAX_LEVELS; i++){
		levels[i] = new Node<DIM, Float>[1 << (DIM * i)];
	}
	
	clear_node_counts<MAX_LEVELS>(node_counts);
	insert_all<DIM, MAX_LEVELS, NODE_THRESHOLD>(n, particles, particlesDst, levels, node_counts);

	return levels;
}

template<size_t DIM, size_t MAX_LEVELS, size_t N_GROUP, typename Float>
std::vector<GroupInfo<DIM, Float, N_GROUP> > groups_from_tree(Node<DIM, Float>* tree[MAX_LEVELS], size_t node_counts[MAX_LEVELS], const Particle<DIM, Float> *particles){
	std::vector<GroupInfo<DIM, Float, N_GROUP> > groups;
	groups.clear();
	//std::cout << "collecting groups:" << std::endl;
	for(size_t level = 0; level < MAX_LEVELS; level++){
		//std::cout << "\t"<< level << std::endl;
		for(size_t node = 0; node < node_counts[level]; node++){
			//std::cout << "\t\t"<< node <<std::endl;
			// We probably don't want to load balance groups because under-full groups can ante-up per-particles
			for(size_t i = 0; i < tree[level][node].childCount; i += N_GROUP){
				//std::cout << "\t\t\t" << i << std::endl;
				GroupInfo<DIM, Float, N_GROUP> group;
				group.childStart = tree[level][node].childStart + i;
				group.childCount = (i < tree[level][node].childCount) ? N_GROUP : (tree[level][node].childCount % N_GROUP);
				group.minX = min_extents(group.childCount, particles + group.childStart);
				group.maxX = max_extents(group.childCount, particles + group.childStart);
				
				// Here, and where we do the same for nodes, we should consider using Miniball instead
				// But this is a faster heuristic and will be fine for plummer spheres (and hopefully in general)
				
				typedef const Particle<DIM, Float>* PointAccessor;
				typedef const Float* CoordAccessor;
				typedef Miniball::Miniball<Miniball::CoordAccessor<PointAccessor, CoordAccessor>> MB;
				MB mb (DIM, particles + group.childStart, particles + group.childStart + group.childCount);
				const Float *center = mb.center();
				for(size_t j = 0; j < DIM; j++){
					group.center.x[j] = center[j];
				}
				group.radius = sqrt(mb.squared_radius());

				groups.push_back(group);
			}
		}
	}
	return groups;
}


// -------------------------


template<size_t DIM, typename Float, size_t PPG>
bool passesMAC(GroupInfo<DIM, Float, PPG> groupInfo, Node<DIM, Float> nodeHere, Float theta) {
	
	Float d = mag<DIM, Float>(groupInfo.center - nodeHere.barycenter) - groupInfo.radius;
	Float l = 2 * nodeHere.radius;
	return d > (l / theta);
	
}

template<size_t DIM, typename Float>
void initNodeStack(Node<DIM, Float>* level, size_t levelCt, std::vector<Node<DIM, Float> > &stack){
	stack = std::vector<Node<DIM, Float> >(level, level + levelCt);
}

template<size_t DIM, typename Float>
void pushAll(Node<DIM, Float>* nodes, size_t nodeCt, std::vector<Node<DIM, Float> > &stack){
	stack.insert(stack.end(), nodes, nodes + nodeCt);
}

// Needs softening
template<size_t DIM, typename Float>
Vec<DIM, Float> calc_force(Float m1, Vec<DIM, Float> v1, Float m2, Vec<DIM, Float> v2, Float softening){
	Vec<DIM, Float> disp = v1 - v2;
	Vec<DIM, Float> force;
	force = disp * ((m1 * m2) / (Float)(softening + pow(mag_sq(disp),1.5)));
	return force;
}

template<size_t DIM, typename Float, TraverseMode Mode>
InteractionType<DIM, Float, Mode> freshInteraction(){
	Vec<DIM, Float> fresh; for(size_t i = 0; i < InteractionElems(Mode, DIM); i++){
		fresh.x[i] = 0;
	}
	return fresh;
}



template<size_t DIM, typename Float, size_t PPG, size_t MAX_LEVELS, TraverseMode Mode>
void traverseTree(size_t nGroups, GroupInfo<DIM, Float, PPG>* groupInfo, size_t startDepth,
				  Node<DIM, Float>* treeLevels[MAX_LEVELS], size_t treeCounts[MAX_LEVELS],
				  Particle<DIM, Float>* particles, InteractionType<DIM, Float, Mode>* interactions, Float softening, Float theta) {
	for (size_t groupI = 0; groupI < nGroups; groupI++) {
		GroupInfo<DIM, Float, PPG> tgInfo = groupInfo[groupI];
		for (size_t particleI = tgInfo.childStart; particleI < tgInfo.childStart + tgInfo.childCount; particleI++){
			
			std::vector<Node<DIM, Float> > currentLevel;
			std::vector<Node<DIM, Float> > nextLevel;
			currentLevel.clear();
			nextLevel.clear();
			initNodeStack(treeLevels[startDepth], treeCounts[startDepth], currentLevel);
			
			Particle<DIM, Float> particle = particles[particleI];

			Vec<DIM, Float> interaction = freshInteraction<DIM, Float, Mode>();
			size_t curDepth = startDepth;
			
			while(currentLevel.size() != 0){
				nextLevel.clear();
				
				size_t startOfs = currentLevel.size();
				while(startOfs > 0){
					size_t toGrab = startOfs - 1;
					//std::cout << "want to grab " << toGrab << " from " << curDepth << " this is " << startOfs << " - 1  = " << (startOfs - 1) << std::endl;
					Node<DIM, Float> nodeHere = currentLevel[toGrab];
					if(passesMAC<DIM, Float, PPG>(tgInfo, nodeHere, theta)){
						// Just interact :)
						InteractionType<DIM, Float, Mode> update = freshInteraction<DIM, Float, Mode>();
						switch (Mode){
							case CountOnly:
								update.x[0] = 1;
								break;
							case HashInteractions:
								update.x[0] = curDepth ^ nodeHere.childCount ^ nodeHere.childStart;
								break;
							case Forces:
								update = calc_force(particle.m, particle.pos, nodeHere.mass, nodeHere.barycenter, softening);
								break;
						}
						interaction = interaction + update;
					} else {
						if(nodeHere.isLeaf){
							for(size_t childI = nodeHere.childStart; childI < nodeHere.childStart + nodeHere.childCount; childI++){
								// Just interact :)
								InteractionType<DIM, Float, Mode> update = freshInteraction<DIM, Float, Mode>();
								switch (Mode){
									case CountOnly:
										update.x[1] = 1; break;
									case HashInteractions:
										update.x[1] = childI; break;
									case Forces:
										update = calc_force(particle.m, particle.pos, nodeHere.mass, nodeHere.barycenter, softening);
										break;
										
								}
								interaction = interaction + update;
							}
						} else {
							//std::cout << "Pushing back " << nodeHere.childCount << " @ " << nodeHere.childStart << std::endl;
							pushAll(treeLevels[curDepth + 1] + nodeHere.childStart, nodeHere.childCount, nextLevel);
						}
					}
					currentLevel.pop_back();
					startOfs--;
				}
				
				std::swap<std::vector<Node<DIM, Float> > >(currentLevel, nextLevel);
				curDepth += 1;
			}
			
			interactions[particleI] = interaction;
		}
	}
	
}


#define DIM 3
#define Float float
#define DT 0.001
#define SOFTENING 0.001
#define THETA 0.5

#define MAX_LEVELS 8
#define NODE_THRESHOLD 16
#define N_GROUP 16
#define TPPB 128
#define INTERACTION_THRESHOLD (TPPB / N_GROUP)

int main(int argc, char* argv[]) {
	int nPs = atoi(argv[1]);
	Particle<DIM,Float> *bodies = new Particle<DIM,Float>[nPs] ;
	Particle<DIM,Float> *bodiesSorted = new Particle<DIM,Float>[nPs] ;
	InteractionType<DIM,Float,Forces> *forces = new InteractionType<DIM,Float,Forces>[nPs];
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
	
	/*
	for(int i = 0; i < 10; i++){
		calc_forces_bruteforce<DIM, Float>(nPs, bodies, forces);
		integrate_system<DIM, Float>(nPs, bodies, forces, DT);
		Float e_now = sys_energy<DIM, Float>(nPs, bodies);
		std::cout << "%dE:\t" << (e_init - e_now) / e_init << std::endl;
	}
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
	std::cout << "We have " << groups.size() << " groups " << std::endl;
	
	//*
	for(int i = 0; i < 10; i++){
		traverseTree<DIM, Float, N_GROUP, MAX_LEVELS, Forces>(groups.size(), groups.data(), 0, tree, node_counts, bodiesSorted, forces, SOFTENING, THETA);
		integrate_system<DIM, Float>(nPs, bodiesSorted, forces, DT);
		Float e_now = sys_energy<DIM, Float>(nPs, bodiesSorted);
		std::cout << "%dE:\t" << (e_init - e_now) / e_init << std::endl;
		
		clear_node_counts<MAX_LEVELS>(node_counts);
		std::swap(bodies, bodiesSorted);
		insert_all<DIM, MAX_LEVELS, NODE_THRESHOLD>(nPs, bodies, bodiesSorted, tree, node_counts);
		groups = groups_from_tree<DIM, MAX_LEVELS, N_GROUP>(tree, node_counts, bodiesSorted);
		
	}
	 //*/
	std::cout << "We have " << groups.size() << " groups " << std::endl;
	
	
	GroupInfoArray<DIM, Float, N_GROUP> gia;
	allocGroupInfoArray(groups.size(), gia);
	for(size_t i = 0; i < groups.size(); i++){
		gia[i] = groups[i];
		//printf("Group info copying to proxy(2): %lu %lu %lu %lu\n",*gia[i].childCount,groups[i].childCount,*gia[i].childStart,groups[i].childStart);
	}
	
	ParticleArray<DIM, Float> pa;
	allocParticleArray(nPs, pa);
	for(size_t i = 0; i < nPs; i++){
		pa[i] = bodiesSorted[i];
	}
	
	VecArray<DIM, Float> va;
	allocVecArray(nPs, va);
	for(size_t i = 0; i < nPs; i++){
		va[i] = forces[i];
	}
	
	NodeArray<DIM, Float> treeA[MAX_LEVELS];
	for(size_t i = 0; i < MAX_LEVELS; i++){
		NodeArray<DIM, Float> level;
		
		allocNodeArray(node_counts[i], level);
		for(size_t j = 0; j < node_counts[i]; j++){
			level[j] = tree[i][j];
			//printf("Node info copying to proxy(2): %lu %lu %lu %lu\n",*level[j].childCount,tree[i][j].childCount,*level[j].childStart,tree[i][j].childStart);
		}
		
		treeA[i] = level;
	}
	
	traverseTreeCUDA<DIM, Float, N_GROUP, MAX_LEVELS, INTERACTION_THRESHOLD, Forces>(groups.size(), gia, 0, treeA, node_counts, nPs, pa, va, SOFTENING, THETA, groups.size(), TPPB);
	
	freeGroupInfoArray(gia);
	freeParticleArray(pa);
	freeVecArray(va);
	for(size_t i = 0; i < MAX_LEVELS; i++){
		freeNodeArray(treeA[i]);
	}
	
	delete_tree<DIM, MAX_LEVELS, Float>(tree);
	
	
	
	
	delete [] bodies;
	delete [] bodiesSorted;
}
