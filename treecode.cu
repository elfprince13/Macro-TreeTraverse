#include "treedefs.h"
#include "CudaArrayCopyUtils.h"
#include "treecodeCU.h"
#include "cudahelper.h"
#include <iostream>

//: We should really be using native CUDA vectors for this.... but that requires more funny typing magic to convert the CPU data

template<size_t DIM, typename Float, size_t PPG> __device__ bool passesMAC(const GroupInfo<DIM, Float, PPG>& groupInfo, const Node<DIM, Float>& nodeHere, Float theta) {
	
	Float d = mag(groupInfo.center - nodeHere.barycenter) - groupInfo.radius;
	Float l = 2 * nodeHere.radius;
	return d > (l / theta);
	
}
/*
 template<template<size_t, typename> class StackElmsArray, size_t DIM, typename Float> __device__ void initStack(StackElmsArray<DIM, Float> level, size_t levelCt, StackElmsArray<DIM, Float> stack, size_t* stackCt, const size_t capacity){
	if(threadIdx.x < levelCt){
 stack[threadIdx.x] = level[threadIdx.x];
	}
	
	if(threadIdx.x == 0){
 *stackCt = levelCt; // We are initializing the stack, so no need to increment previous value
 //printf("%d.%d: %lu = %lu\n", blockIdx.x, threadIdx.x, *stackCt, levelCt);
	}
	
 }
 */

template<size_t DIM, typename Float> __device__ void initParticleStack(ParticleArray<DIM, Float> level, size_t levelCt, ParticleArray<DIM, Float> stack, size_t* stackCt){
	//*
	if(threadIdx.x < levelCt){
				printf("%d.%d Copying to stack %lu (should be <= %lu) nodes from %p to %p. stackCt @ %d / %lu\n",blockIdx.x, threadIdx.x,levelCt,level.elems,level.m,stack.m,threadIdx.x,stack.elems);
		/*if(threadIdx.x == 0 && blockIdx.x == 0){
			printf("%d Copied particle (%f) to (%f)\n", blockIdx.x, stack.m[threadIdx.x], level.m[threadIdx.x]);
		}*/
		Particle<DIM, Float> pHere;
		level.get(threadIdx.x, pHere);
		stack.set(threadIdx.x, pHere);
	}
	//*/
	//*
	if(threadIdx.x == 0){
		*stackCt = levelCt; // We are initializing the stack, so no need to increment previous value
							//printf("%d.%d: %lu = %lu\n", blockIdx.x, threadIdx.x, *stackCt, levelCt);
	}
	//*/
}

// Specialization seems broken here :(
template<size_t DIM, typename Float> __device__ void initNodeStack(NodeArray<DIM, Float> level, size_t levelCt, NodeArray<DIM, Float> stack, size_t* stackCt){
	//*
	if(threadIdx.x < levelCt){
		printf("%d.%d Copying to stack %lu (should be <= %lu) nodes from %p to %p. stackCt @ %d / %lu\n",blockIdx.x, threadIdx.x,levelCt,level.elems,level.childCount,stack.childCount,threadIdx.x,stack.elems);
		
		Node<DIM, Float> nHere;
		level.get(threadIdx.x, nHere);
		stack.set(threadIdx.x, nHere);
		/*if(threadIdx.x == 0 && blockIdx.x == 0){
			printf("%d Copied node (%lu %lu) to (%lu %lu)\n", blockIdx.x, stack.childCount[threadIdx.x], stack.childStart[threadIdx.x], level.childCount[threadIdx.x],level.childStart[threadIdx.x]);
		}*/
	}
	//*/
	//*
	if(threadIdx.x == 0){
		*stackCt = levelCt; // We are initializing the stack, so no need to increment previous value
							//printf("%d.%d: %lu = %lu\n", blockIdx.x, threadIdx.x, *stackCt, levelCt);
	}
	//*/
}

template<size_t DIM, typename Float> __device__ void dumpStackChildren(NodeArray<DIM, Float> stack, const size_t* stackCt){
	size_t dst = *stackCt;//atomicAdd((unsigned long long*)stackCt, (unsigned long long)0);
	for(size_t i = 0; i < dst; i++){
			printf("(%lu) see (%lu %lu) in (%p/%lu)\n",i,stack.childStart[i],stack.childCount[i], stackCt, dst);
	}
}

template<size_t DIM, typename Float> __device__ void pushAllNodes(const NodeArray<DIM, Float> nodes, const size_t nodeCt, NodeArray<DIM, Float> stack, size_t* stackCt){
	// This is a weird compiler bug. There's no reason this shouldn't have worked without the cast.
	size_t dst = atomicAdd((unsigned long long*)stackCt, (unsigned long long)nodeCt);
	printf("%d.%d Pushing to stack %lu (should be <= %lu) nodes from %p to %p. stackCt @ %lu / %lu\n",blockIdx.x, threadIdx.x,nodeCt,nodes.elems,nodes.childCount,stack.childCount,dst,stack.elems);
	for(size_t i = dst, j = 0; i < dst + nodeCt; i++, j++){
		Node<DIM, Float> nHere;
		nodes.get(j, nHere);
		stack.set(i, nHere);
		/*if(blockIdx.x == 0){
			printf("%d.%d Pushed node (%lu %lu) to (%lu %lu) with (%lu,%lu)\n",
				   blockIdx.x,threadIdx.x, stack.childCount[i], stack.childStart[i], nodes.childCount[j],nodes.childStart[j],i,j);
		}*/
	}
}

template<size_t DIM, typename Float> __device__ void pushAllParticles(const ParticleArray<DIM, Float> nodes, const size_t nodeCt, ParticleArray<DIM, Float> stack, size_t* stackCt){

	// This is a weird compiler bug. There's no reason this shouldn't have worked without the cast.
	size_t dst = atomicAdd((unsigned long long*)stackCt, (unsigned long long)nodeCt);
	printf("%d.%d Pushing to stack %lu (should be <= %lu) particles from %p to %p. stackCt @ %lu / %lu\n",blockIdx.x, threadIdx.x,nodeCt,nodes.elems,nodes.m,stack.m,dst,stack.elems);
	for(size_t i = dst, j = 0; i < dst+ nodeCt; i++, j++){
		/*if(blockIdx.x == 0){
			printf("%d.%d Pushing particle (%f) to (%f) with (%lu,%lu)\n", blockIdx.x,threadIdx.x, stack.m[i], nodes.m[j],i,j);
		}*/
		Particle<DIM, Float> pHere;
		nodes.get(j, pHere);
		stack.set(i, pHere);
	}
}

// Needs softening
template<size_t DIM, typename Float> __device__ Vec<DIM, Float> calc_force(Float m1, Vec<DIM, Float> v1, Float m2, Vec<DIM, Float> v2, Float softening){
	Vec<DIM, Float> disp = v1 - v2;
	Vec<DIM, Float> force;
	//Another CUDA-induced casting hack here. Otherwise it tries to call the device version of the code
	force = disp * ((m1 * m2) / (Float)(softening + pow((Float)mag_sq(disp),(Float)1.5)));
	return force;
}

template<size_t DIM, typename Float, TraverseMode Mode> __device__ InteractionType<DIM, Float, Mode> freshInteraction(){
	InteractionType<DIM, Float, Mode> fresh; for(size_t i = 0; i < DIM; i++){
		fresh.x[i] = 0.0;
	}
	return fresh;
}

template<typename T> __device__ inline void swap(T& a, T& b){
	T c(a); a=b; b=c;
}

template<size_t DIM, typename Float, size_t PPG, size_t MAX_LEVELS, size_t INTERACTION_THRESHOLD, TraverseMode Mode>
__global__ void traverseTreeKernel(const size_t nGroups, const GroupInfoArray<DIM, Float, PPG> groupInfo,
								   const size_t startDepth, NodeArray<DIM, Float>* treeLevels, const size_t* treeCounts,
								   const size_t n, const ParticleArray<DIM, Float> particles, const InteractionTypeArray<DIM, Float, Mode> interactions,
								   const Float softening, const Float theta,
								   size_t *bfsStackCounters, NodeArray<DIM, Float> bfsStackBuffers, const size_t stackCapacity) {
	
	if(blockIdx.x == 0 && threadIdx.x == 0){
		printf("Validating %lu groups @ (%p %p)\n", nGroups, groupInfo.childStart, groupInfo.childCount);
		for(size_t i = 0; i < nGroups; i++){
			printf("\t(%lu %lu)",groupInfo.childStart[i], groupInfo.childCount[i]);
		} printf("\n\n");
		printf("Validating tree\n");
		for(size_t i = 0; i < MAX_LEVELS; i++){
			printf("Layer %lu: has %lu @ (%p %p %p)\n", i, treeCounts[i], treeLevels[i].childStart, treeLevels[i].childCount, treeLevels[i].isLeaf);
			for(size_t j = 0; j < treeCounts[i]; j++){
				printf("\t(%lu %lu %d)",treeLevels[i].childStart[j], treeLevels[i].childCount[j], treeLevels[i].isLeaf[j]);
			} printf("\n\n");
		} printf("\n");
	}
	
	/*
	__threadfence();
	__syncthreads();
	if(blockIdx.x == 0 && threadIdx.x == 0){
	
		printf("Validating particles\n");
		printf("%p ",particles.m);
		for(size_t j = 0; j < DIM; j++){
			printf("%p ", particles.pos.x[j]);
		}
		for(size_t j = 0; j < DIM; j++){
			printf("%p ", particles.vel.x[j]);
		} printf("\n(");
		for(size_t i = 0; i < n; i++){
			printf("(%f ",particles.m[i]);
			for(size_t j = 0; j < DIM; j++){
				printf("%f ", particles.pos[i].x[j]);
			}
			for(size_t j = 0; j < DIM; j++){
				printf("%f ", particles.vel[i].x[j]);
			} printf(")\t");
		}printf("\n\n");
	
	}
	 //*/
	
	__threadfence();
	__syncthreads();
	
	if(blockIdx.x == 0 && threadIdx.x == 0) dumpStackChildren(treeLevels[1], treeCounts+1);
	__threadfence_block();
	__syncthreads();
	
	
	__shared__ size_t interactionCounters[2];
	__shared__ Float particleMass[32*INTERACTION_THRESHOLD];
	__shared__ Float particlePos[DIM * 32*INTERACTION_THRESHOLD];
	__shared__ Float particleVel[DIM * 32*INTERACTION_THRESHOLD];
	
	__shared__ bool nodeLeaves[32*INTERACTION_THRESHOLD];
	__shared__ size_t nodeChildCount[32*INTERACTION_THRESHOLD];
	__shared__ size_t nodeChildStart[32*INTERACTION_THRESHOLD];
	__shared__ Float nodeMinX[3*32*INTERACTION_THRESHOLD];
	__shared__ Float nodeMaxX[3*32*INTERACTION_THRESHOLD];
	__shared__ Float nodeBarycenter[3*32*INTERACTION_THRESHOLD];
	__shared__ Float nodeMass[32*INTERACTION_THRESHOLD];
	__shared__ Float nodeRadius[32*INTERACTION_THRESHOLD];
	
	
	if(threadIdx.x == 0 && blockIdx.x == 0){
		printf("%p %p %p %p\n",interactionCounters, particleMass, particlePos, particleVel);
		printf("%p %p %p\n", nodeLeaves, nodeChildCount, nodeChildStart);
		printf("%p %p %p\n", nodeMinX, nodeMaxX, nodeBarycenter);
		printf("%p %p\n", nodeMass, nodeRadius);
	}
	
	
	ParticleArray<DIM, Float> particleInteractionList;
	particleInteractionList.m = particleMass;
	for(size_t j = 0; j < DIM; j++){
		particleInteractionList.pos.x[j] = particlePos + (j * 2 * INTERACTION_THRESHOLD);
	}
	for(size_t j = 0; j < DIM; j++){
		particleInteractionList.vel.x[j] = particleVel + (j * 2 * INTERACTION_THRESHOLD);
	}
	particleInteractionList.setCapacity(32*INTERACTION_THRESHOLD);
	
	if(threadIdx.x == 0 && blockIdx.x == 0){
		printf("%p\n",particleInteractionList.m);
		for(size_t j = 0; j < DIM; j++){
			printf("%p ", particleInteractionList.pos.x[j]);
		} printf("\n");
		for(size_t j = 0; j < DIM; j++){
			printf("%p ", particleInteractionList.vel.x[j]);
		} printf("\n");
	}
	
	
	NodeArray<DIM, Float> nodeInteractionList;
	nodeInteractionList.isLeaf = nodeLeaves;
	nodeInteractionList.childCount = nodeChildCount;
	nodeInteractionList.childStart = nodeChildStart;
	for(size_t j = 0; j < DIM; j++){
		nodeInteractionList.minX.x[j] = nodeMinX + (j * 2 * INTERACTION_THRESHOLD);
	}
	for(size_t j = 0; j < DIM; j++){
		nodeInteractionList.maxX.x[j] = nodeMaxX + (j * 2 * INTERACTION_THRESHOLD);
	}
	for(size_t j = 0; j < DIM; j++){
		nodeInteractionList.barycenter.x[j] = nodeBarycenter + (j * 2 * INTERACTION_THRESHOLD);
	}
	nodeInteractionList.mass = nodeMass;
	nodeInteractionList.radius = nodeRadius;
	nodeInteractionList.setCapacity(32*INTERACTION_THRESHOLD);
	

	if(threadIdx.x == 0 && blockIdx.x == 0){
		printf("%p %p %p\n",nodeInteractionList.isLeaf, nodeInteractionList.childCount, nodeInteractionList.childStart);
		for(size_t j = 0; j < DIM; j++){
			printf("%p ", nodeInteractionList.minX.x[j]);
		} printf("\n");
		for(size_t j = 0; j < DIM; j++){
			printf("%p ", nodeInteractionList.maxX.x[j]);
		} printf("\n");
		for(size_t j = 0; j < DIM; j++){
			printf("%p ", nodeInteractionList.barycenter.x[j]);
		} printf("\n");
		printf("%p %p\n",nodeInteractionList.mass, nodeInteractionList.radius);
	}
	
	
	if(blockIdx.x >= nGroups) return; // This probably shouldn't happen?
	else {
		GroupInfo<DIM, Float, PPG> tgInfo;
		groupInfo.get(blockIdx.x,tgInfo);
		int threadsPerPart = blockDim.x / tgInfo.childCount;
		
		
		size_t* pGLCt = interactionCounters;
		ParticleArray<DIM, Float> pGList = particleInteractionList;
		ParticleArray<DIM, Float> dummyP;
		initParticleStack(dummyP, 0, pGList, pGLCt);
		__threadfence_block();
		
		if(threadIdx.x == 0 && blockIdx.x == 0){
			printf("%p %p\n",pGLCt, pGList.m);
			for(size_t j = 0; j < DIM; j++){
				printf("%p ", pGList.pos.x[j]);
			} printf("\n");
			for(size_t j = 0; j < DIM; j++){
				printf("%p ", pGList.vel.x[j]);
			} printf("\n");
		}


		
		
		size_t* nGLCt = interactionCounters + 1;
		NodeArray<DIM, Float> nGList = nodeInteractionList;
		NodeArray<DIM, Float> dummyN;
		initNodeStack(dummyN, 0, nGList, nGLCt);
		__threadfence_block();
		
		if(threadIdx.x == 0 && blockIdx.x == 0){
			printf("%p %p %p %p\n",nGLCt, nGList.isLeaf, nGList.childCount, nGList.childStart);
			for(size_t j = 0; j < DIM; j++){
				printf("%p ", nGList.minX.x[j]);
			} printf("\n");
			for(size_t j = 0; j < DIM; j++){
				printf("%p ", nGList.maxX.x[j]);
			} printf("\n");
			for(size_t j = 0; j < DIM; j++){
				printf("%p ", nGList.barycenter.x[j]);
			} printf("\n");
			printf("%p %p\n",nGList.mass, nGList.radius);
		}
		
		size_t* cLCt = bfsStackCounters + 2 * blockIdx.x;
		NodeArray<DIM, Float> currentLevel = bfsStackBuffers + 2 * blockIdx.x * stackCapacity;
		currentLevel.setCapacity(stackCapacity);
		__threadfence_block();
		if(threadIdx.x == 0 && blockIdx.x == 0){
			printf("%p %p %p %p\n",cLCt, currentLevel.isLeaf, currentLevel.childCount, currentLevel.childStart);
			for(size_t j = 0; j < DIM; j++){
				printf("%p ", currentLevel.minX.x[j]);
			} printf("\n");
			for(size_t j = 0; j < DIM; j++){
				printf("%p ", currentLevel.maxX.x[j]);
			} printf("\n");
			for(size_t j = 0; j < DIM; j++){
				printf("%p ", currentLevel.barycenter.x[j]);
			} printf("\n");
			printf("%p %p\n",currentLevel.mass, currentLevel.radius);
		}
		
		
		if(threadIdx.x == 0 && blockIdx.x == 0){
			printf("%d initing stack: %lu \n", blockIdx.x, startDepth);
			printf("%d continues: %p %p\n",blockIdx.x,treeLevels[startDepth].childCount, treeLevels[startDepth].childStart);
			printf("%d contains: %lu %lu\n",blockIdx.x,treeLevels[startDepth].childCount[0], treeLevels[startDepth].childStart[0]);
			//treeLevels[startDepth].childCount[0], treeLevels[startDepth].childStart[0]
		}
		initNodeStack(treeLevels[startDepth], treeCounts[startDepth], currentLevel, cLCt);
		
		
		if(blockIdx.x == 0 && threadIdx.x == 0)dumpStackChildren(currentLevel, cLCt);
		__threadfence_block();
		__syncthreads();
		
		size_t* nLCt = bfsStackCounters + 2 * blockIdx.x + 1;
		NodeArray<DIM, Float> nextLevel = bfsStackBuffers + (2 * blockIdx.x + 1) * stackCapacity;
		nextLevel.setCapacity(stackCapacity);
		if(threadIdx.x == 0 && blockIdx.x == 0){
			printf("%p %p %p %p\n",nLCt, nextLevel.isLeaf, nextLevel.childCount, nextLevel.childStart);
			for(size_t j = 0; j < DIM; j++){
				printf("%p ", nextLevel.minX.x[j]);
			} printf("\n");
			for(size_t j = 0; j < DIM; j++){
				printf("%p ", nextLevel.maxX.x[j]);
			} printf("\n");
			for(size_t j = 0; j < DIM; j++){
				printf("%p ", nextLevel.barycenter.x[j]);
			} printf("\n");
			printf("%p %p\n",nextLevel.mass, nextLevel.radius);
		}
		
			
		__syncthreads();
		if(threadIdx.x == 0 && blockIdx.x == 0){
			printf("%d post init stack: %lu \n", blockIdx.x, startDepth);
			printf("%d continues: %p %p\n",blockIdx.x,currentLevel.childCount, currentLevel.childStart);
			printf("%d contains: %lu %lu\n",blockIdx.x,currentLevel.childCount[0], currentLevel.childStart[0]);
			//treeLevels[startDepth].childCount[0], treeLevels[startDepth].childStart[0]
		}
		__syncthreads();
		
		
		
		Particle<DIM, Float> particle;
		if(threadIdx.x > threadsPerPart * tgInfo.childCount){
			particles.get(tgInfo.childStart + (threadIdx.x % tgInfo.childCount), particle);
		}
		
		InteractionType<DIM, Float, Mode> interaction = freshInteraction<DIM, Float, Mode>();
		size_t curDepth = startDepth;
		while(*cLCt != 0 ){//&& curDepth < MAX_LEVELS){ // Second condition shouldn't matter....
			if(threadIdx.x == 0 && blockIdx.x == 0)printf("Entering the land of disturbing loops\n");
			if(threadIdx.x == 0){
				*nLCt = 0;
			}
			if(blockIdx.x == 0 && threadIdx.x == 0) dumpStackChildren(currentLevel, cLCt);
			__threadfence_block();
			__syncthreads();
			if(blockIdx.x == 0 && threadIdx.x == 0)dumpStackChildren(nextLevel, nLCt);
			__threadfence_block();
			__syncthreads();
			
			if(threadIdx.x == 0 && blockIdx.x == 0){
				printf("%p %p %p %p\n",cLCt, currentLevel.isLeaf, currentLevel.childCount, currentLevel.childStart);
				for(size_t j = 0; j < DIM; j++){
					printf("%p ", currentLevel.minX.x[j]);
				} printf("\n");
				for(size_t j = 0; j < DIM; j++){
					printf("%p ", currentLevel.maxX.x[j]);
				} printf("\n");
				for(size_t j = 0; j < DIM; j++){
					printf("%p ", currentLevel.barycenter.x[j]);
				} printf("\n");
				printf("%p %p\n",currentLevel.mass, currentLevel.radius);
				printf("%p %p %p %p\n",nLCt, nextLevel.isLeaf, nextLevel.childCount, nextLevel.childStart);
				for(size_t j = 0; j < DIM; j++){
					printf("%p ", nextLevel.minX.x[j]);
				} printf("\n");
				for(size_t j = 0; j < DIM; j++){
					printf("%p ", nextLevel.maxX.x[j]);
				} printf("\n");
				for(size_t j = 0; j < DIM; j++){
					printf("%p ", nextLevel.barycenter.x[j]);
				} printf("\n");
				printf("%p %p\n",nextLevel.mass, nextLevel.radius);
			}
			
			__threadfence_block();
			__syncthreads();
			
			ptrdiff_t startOfs = *cLCt;
			while(startOfs > 0){
				if(threadIdx.x == 0 && blockIdx.x == 0) printf("\tEntering the inner crazy loop\n");
				ptrdiff_t toGrab = startOfs - blockDim.x + threadIdx.x;
				if(toGrab >= 0){
					Node<DIM, Float> nodeHere;
					currentLevel.get(toGrab, nodeHere);
					if(toGrab == 0 &&
					   blockIdx.x == 0){
						printf("%d.%d @ %lu:\t%lu %lu vs %lu %lu with %lu %ld \n", blockIdx.x, threadIdx.x, curDepth, nodeHere.childStart, nodeHere.childCount, currentLevel.childStart[toGrab], currentLevel.childCount[toGrab], *cLCt, toGrab);
					}
					//*
					if(passesMAC(tgInfo, nodeHere, theta)){
						if(blockIdx.x == 0) printf("\t%d accepted MAC\n",threadIdx.x);
						if(INTERACTION_THRESHOLD > 0){
							// Store to C/G list
							if(blockIdx.x == 0) printf("\t%d found the following:\n\t(%lu %lu) @ (%p/%lu), writing to stack at pos %lu @ %p\n", threadIdx.x, nodeHere.childStart, nodeHere.childCount, treeLevels[curDepth + 1].childStart, curDepth + 1, *nLCt,nLCt);
							
							NodeArray<DIM, Float> tmpArray(nodeHere);
							size_t tmpCt = 1;
							
							pushAllNodes(tmpArray, tmpCt, nGList, nGLCt);
						} else if(threadIdx.x < tgInfo.childCount){
							//interaction = interaction + calc_force(particle.m, particle.pos, nodeHere.mass, nodeHere.barycenter, softening);
						}
					} else {
						if(blockIdx.x == 0) printf("\t%d rejected MAC\n",threadIdx.x);
						if(nodeHere.isLeaf){
							if(INTERACTION_THRESHOLD > 0){
								// Store to P/G list
								//printf("Pushing particles %lu particles, ")
								pushAllParticles(particles + nodeHere.childStart, nodeHere.childCount, pGList, pGLCt);
							} else {
								/*
								 for(size_t pI = nodeHere.childCount; pI > 0; pI -= threadsPerPart ){
									ptrdiff_t toGrab = pI - threadsPerPart + (threadIdx.x / tgInfo.childCount);
									if(toGrab >= 0){
								 interaction = interaction + calc_force(particle.m, particle.pos, particles[nodeHere.childStart + toGrab].m, particles[nodeHere.childStart + toGrab].pos, softening);
									}
								 }
								 */
							}
						} else {
							//if(curDepth + 1 < MAX_LEVELS && nodeHere.childStart < treeCounts[curDepth + 1]){
							if(blockIdx.x == 0) printf("\t%d found the following:\n\t(%lu %lu) @ (%p/%lu), writing to stack at pos %lu @ %p\n", threadIdx.x, nodeHere.childStart, nodeHere.childCount, treeLevels[curDepth + 1].childStart, curDepth + 1, *nLCt,nLCt);
							
							
							if(blockIdx.x == 0)dumpStackChildren(treeLevels[curDepth+1], treeCounts+curDepth+1);
							if(blockIdx.x == 0)dumpStackChildren(nextLevel, nLCt);
							
							
							pushAllNodes(treeLevels[curDepth + 1] + nodeHere.childStart, nodeHere.childCount, nextLevel, nLCt);
							__threadfence_block();
							if(blockIdx.x == 0)dumpStackChildren(treeLevels[curDepth+1], treeCounts+curDepth+1);
							if(blockIdx.x == 0)dumpStackChildren(nextLevel, nLCt);
							
							//}
						}
					}
					//*/
				}
				__threadfence_block();
				__syncthreads();
				
				
				if(threadIdx.x == 0 && blockIdx.x == 0){
					printf("%p %p %p %p\n",cLCt, currentLevel.isLeaf, currentLevel.childCount, currentLevel.childStart);
					for(size_t j = 0; j < DIM; j++){
						printf("%p ", currentLevel.minX.x[j]);
					} printf("\n");
					for(size_t j = 0; j < DIM; j++){
						printf("%p ", currentLevel.maxX.x[j]);
					} printf("\n");
					for(size_t j = 0; j < DIM; j++){
						printf("%p ", currentLevel.barycenter.x[j]);
					} printf("\n");
					printf("%p %p\n",currentLevel.mass, currentLevel.radius);
					printf("%p %p %p %p\n",nLCt, nextLevel.isLeaf, nextLevel.childCount, nextLevel.childStart);
					for(size_t j = 0; j < DIM; j++){
						printf("%p ", nextLevel.minX.x[j]);
					} printf("\n");
					for(size_t j = 0; j < DIM; j++){
						printf("%p ", nextLevel.maxX.x[j]);
					} printf("\n");
					for(size_t j = 0; j < DIM; j++){
						printf("%p ", nextLevel.barycenter.x[j]);
					} printf("\n");
					printf("%p %p\n",nextLevel.mass, nextLevel.radius);
				}
				
				
				//*
				if(INTERACTION_THRESHOLD > 0){ // Can't diverge, compile-time constant
					ptrdiff_t innerStartOfs;
					for(innerStartOfs = *nGLCt; innerStartOfs >= INTERACTION_THRESHOLD; innerStartOfs -= threadsPerPart){
						ptrdiff_t toGrab = innerStartOfs - threadsPerPart + (threadIdx.x / tgInfo.childCount);
						if(toGrab >= 0){
							Node<DIM, Float> nHere;
							nGList.get(toGrab, nHere);
							interaction = interaction + calc_force(particle.m, particle.pos, nHere.mass, nHere.barycenter, softening);
						}
					}
					// Need to update stack pointer
					if(threadIdx.x == 0){
						*nGLCt = (innerStartOfs < 0) ? 0 : innerStartOfs;
					}
				 
					for(innerStartOfs = *pGLCt; innerStartOfs >= INTERACTION_THRESHOLD; innerStartOfs -= threadsPerPart){
						ptrdiff_t toGrab = innerStartOfs - threadsPerPart + (threadIdx.x / tgInfo.childCount);
						if(toGrab >= 0){
							Particle<DIM, Float> pHere;
							pGList.get(toGrab, pHere);
							interaction = interaction + calc_force(particle.m, particle.pos, pHere.m, pHere.pos, softening);
						}
					}
					// Need to update stack pointer
					// Need to update stack pointer
					if(threadIdx.x == 0){
						*pGLCt = (innerStartOfs < 0) ? 0 : innerStartOfs;
					}
				 
				}
				//*/
				
				if(threadIdx.x == 0 && blockIdx.x == 0)printf("Try going around again\n");
				
				
				startOfs -= blockDim.x;
				
				
			}
			
			if(threadIdx.x == 0 && blockIdx.x == 0) printf("Done inside: %lu work remaining\n",*nLCt);
			
			swap<NodeArray<DIM, Float>>(currentLevel, nextLevel);
			swap<size_t*>(cLCt, nLCt);
			curDepth += 1;
		}
		
		// Process remaining interactions and reduce if multithreading in play
		
		//*/
		
	}
	
}


// Something is badly wrong with template resolution if we switch to InteractionType here.
// I think the compilers are doing name-mangling differently or something
template<size_t DIM, typename Float, size_t PPG, size_t MAX_LEVELS, size_t INTERACTION_THRESHOLD, TraverseMode Mode>
void traverseTreeCUDA(size_t nGroups, GroupInfoArray<DIM, Float, PPG> groupInfo, size_t startDepth,
					  NodeArray<DIM, Float> treeLevels[MAX_LEVELS], size_t treeCounts[MAX_LEVELS], size_t n, ParticleArray<DIM, Float> particles, VecArray<DIM, Float> interactions, Float softening, Float theta, size_t blockCt, size_t threadCt){
	
	NodeArray<DIM, Float> placeHolderLevels[MAX_LEVELS];
	makeDeviceTree<DIM, Float, MAX_LEVELS>(treeLevels, placeHolderLevels, treeCounts);
	NodeArray<DIM, Float>* cuTreeLevels;
	gpuErrchk( (cudaMalloc(&cuTreeLevels, MAX_LEVELS*sizeof(NodeArray<DIM, Float>))) );
	gpuErrchk( (cudaMemcpy(cuTreeLevels, placeHolderLevels, MAX_LEVELS*sizeof(NodeArray<DIM, Float>), cudaMemcpyHostToDevice)) );
	
	size_t* cuTreeCounts;
	gpuErrchk( (cudaMalloc(&cuTreeCounts, MAX_LEVELS * sizeof(size_t))) );
	gpuErrchk( (cudaMemcpy(cuTreeCounts, treeCounts, MAX_LEVELS * sizeof(size_t), cudaMemcpyHostToDevice)) );
	
	
	size_t biggestRow = 0;
	for(size_t level = 0; level < MAX_LEVELS; level++){
		biggestRow = (treeCounts[level] > biggestRow) ? treeCounts[level] : biggestRow;
	}
	
	const size_t stackCapacity = biggestRow;
	NodeArray<DIM, Float> bfsStackBuffers;
	size_t * bfsStackCounters;
	allocDeviceNodeArray(blockCt * 2 * stackCapacity, bfsStackBuffers);
	gpuErrchk( (cudaMalloc(&bfsStackCounters, blockCt * 2 * sizeof(size_t))) );
	
	
	GroupInfoArray<DIM, Float, PPG> cuGroupInfo;
	allocDeviceGroupInfoArray(nGroups, cuGroupInfo);
	copyDeviceGroupInfoArray(nGroups, cuGroupInfo, groupInfo, cudaMemcpyHostToDevice);
	
	ParticleArray<DIM, Float> cuParticles;
	allocDeviceParticleArray(n, cuParticles);
	copyDeviceParticleArray(n, cuParticles, particles, cudaMemcpyHostToDevice);
	
	InteractionTypeArray<DIM, Float, Mode> cuInteractions;
	allocDeviceVecArray(n, cuInteractions);
	copyDeviceVecArray(n, cuInteractions, interactions, cudaMemcpyHostToDevice);
	
	dim3 dimGrid(blockCt);
	dim3 dimBlock(threadCt);
	std::cout << "Trying to launch with " << threadCt << " / block with " << blockCt << " blocks" << std::endl;
	
	traverseTreeKernel<DIM, Float, PPG, MAX_LEVELS, INTERACTION_THRESHOLD, Mode><<<dimGrid, dimBlock>>>(nGroups, cuGroupInfo, startDepth, cuTreeLevels, cuTreeCounts, n, cuParticles, cuInteractions, softening, theta, bfsStackCounters, bfsStackBuffers, stackCapacity);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	
	copyDeviceVecArray(n, interactions, cuInteractions, cudaMemcpyDeviceToHost);
	
	freeDeviceVecArray(cuInteractions);
	freeDeviceParticleArray(cuParticles);
	freeDeviceGroupInfoArray(cuGroupInfo);
	freeDeviceNodeArray(bfsStackBuffers);
	gpuErrchk( (cudaFree(bfsStackCounters)) );
	gpuErrchk( (cudaFree(cuTreeCounts)) );
	freeDeviceTree<DIM, Float, MAX_LEVELS>(placeHolderLevels);
	gpuErrchk( (cudaFree(cuTreeLevels)) );
	
	
	
}

template void traverseTreeCUDA<3, float, 16, 8, 8, Forces>(size_t, GroupInfoArray<3, float, 16>, size_t, NodeArray<3, float> *, size_t *, size_t, ParticleArray<3, float>, InteractionTypeArray<3, float, Forces>, float, float, size_t, size_t);


