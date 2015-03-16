#include "treedefs.h"
#include "treecodeCU.h"


//: We should really be using native CUDA vectors for this.... but that requires more funny typing magic to convert the CPU data

template<size_t DIM, typename Float, size_t PPG> __device__ bool passesMAC(GroupInfo<DIM, Float, PPG> groupInfo, Node<DIM, Float> nodeHere, Float theta) {
	
	Float d = mag(groupInfo.center - nodeHere.barycenter) - groupInfo.radius;
	Float l = 2 * nodeHere.radius;
	return d > (l / theta);
	
}

template<template<size_t, typename> class StackElms, size_t DIM, typename Float> __device__ void initStack(StackElms<DIM, Float>* level, size_t levelCt, StackElms<DIM, Float>* stack, size_t* stackCt, const size_t capacity){
	if(threadIdx.x < levelCt){
		stack[threadIdx.x] = level[threadIdx.x];
	}
	if(threadIdx.x == 0){
		*stackCt = levelCt; // We are initializing the stack, so no need to increment previous value
	}
}

template<template<size_t, typename> class StackElms, size_t DIM, typename Float> __device__ void pushAll(StackElms<DIM, Float>* nodes, size_t nodeCt, StackElms<DIM, Float>* stack, size_t* stackCt){
	// This is a weird compiler bug. There's no reason this shouldn't have worked without the cast.
	size_t dst = atomicAdd((unsigned long long*)stackCt, (unsigned long long)nodeCt);
	for(size_t i = dst, j = 0; i < dst+ nodeCt; i++, j++){
		stack[i] = nodes[j];
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
__global__ void traverseTreeKernel(size_t nGroups, GroupInfo<DIM, Float, PPG>* groupInfo, size_t startDepth,
							 Node<DIM, Float>* treeLevels[MAX_LEVELS], size_t treeCounts[MAX_LEVELS], Particle<DIM, Float>* particles, Vec<DIM, Float>* interactions, Float softening, Float theta, unsigned char *bfsStackBuffers, const size_t stackCapacity) {
	__shared__ unsigned char smem[2 * sizeof(size_t) + 2 * INTERACTION_THRESHOLD * (sizeof(Node<DIM, Float>) + sizeof(Particle<DIM, Float>))];
	size_t* pGLCt = (size_t*)smem;
	Particle<DIM, Float>* pGList = (Particle<DIM, Float>*)(smem + sizeof(size_t));
	initStack((Particle<DIM, Float>*)nullptr, 0, pGList, pGLCt, 2 * INTERACTION_THRESHOLD);
	
	size_t* nGLCt = (size_t*)(smem + sizeof(size_t) + 2 * INTERACTION_THRESHOLD * sizeof(Particle<DIM, Float>));
	Node<DIM, Float>* nGList = (Node<DIM, Float>*)(smem + 2 * (sizeof(size_t) + INTERACTION_THRESHOLD * sizeof(Particle<DIM, Float>)));
	initStack((Node<DIM, Float>*)nullptr, 0, nGList, nGLCt, 2 * INTERACTION_THRESHOLD);
	
	
	if(blockIdx.x >= nGroups) return; // This probably shouldn't happen?
	else {
		GroupInfo<DIM, Float, PPG> tgInfo = groupInfo[blockIdx.x];
		int threadsPerPart = blockDim.x / tgInfo.childCount;
		if(threadIdx.x > threadsPerPart * tgInfo.childCount) return; // Really shouldn't do this with __syncthreads!
		else {
			const size_t stackBytes = sizeof(size_t) + stackCapacity * sizeof(Node<DIM, Float>);
			// We should forget about the rest of the buffer:
			bfsStackBuffers = bfsStackBuffers + 2 * stackBytes * blockIdx.x;
			size_t* cLCt = (size_t*)bfsStackBuffers;
			Node<DIM, Float>* currentLevel = (Node<DIM, Float>*)(bfsStackBuffers + sizeof(size_t));
			initStack(treeLevels[startDepth], treeCounts[startDepth], currentLevel, cLCt, stackCapacity);
			
			size_t* nLCt = (size_t*)(bfsStackBuffers + stackBytes);
			Node<DIM, Float>* nextLevel = (Node<DIM, Float>*)(bfsStackBuffers + stackBytes + sizeof(size_t));
			
			__syncthreads();
			
			Particle<DIM, Float> particle = particles[tgInfo.childStart + (threadIdx.x % tgInfo.childCount)];
			
			InteractionType<DIM, Float, Mode> interaction = freshInteraction<DIM, Float, Mode>();
			size_t curDepth = startDepth;
			
			while(*cLCt != 0){
				if(threadIdx.x == 0){
					*nLCt = 0;
				}
				__syncthreads();
			
				size_t startOfs = *cLCt;
				while(startOfs > 0){
					ptrdiff_t toGrab = startOfs - blockDim.x + threadIdx.x;
					if(toGrab >= 0){
						Node<DIM, Float> nodeHere = currentLevel[toGrab];
						if(passesMAC(tgInfo, nodeHere, theta)){
							if(INTERACTION_THRESHOLD > 0){
								// Store to C/G list
								pushAll(&nodeHere, 1, nGList, nGLCt);
							} else if(threadIdx.x < tgInfo.childCount){
								interaction = interaction + calc_force(particle.m, particle.pos, nodeHere.mass, nodeHere.barycenter, softening);
							}
						} else {
							if(nodeHere.isLeaf){
								if(INTERACTION_THRESHOLD > 0){
									// Store to P/G list
									pushAll(particles + nodeHere.childStart, nodeHere.childCount, pGList, pGLCt);
								} else {
									for(size_t pI = nodeHere.childCount; pI > 0; pI -= threadsPerPart ){
										ptrdiff_t toGrab = pI - threadsPerPart + (threadIdx.x / tgInfo.childCount);
										if(toGrab >= 0){
											interaction = interaction + calc_force(particle.m, particle.pos, particles[nodeHere.childStart + toGrab].m, particles[nodeHere.childStart + toGrab].pos, softening);
										}
									}
								}
							} else {
								pushAll(treeLevels[curDepth + 1] + nodeHere.childStart, nodeHere.childCount, nextLevel, nLCt);
							}
						}
					}
					__syncthreads();
					if(INTERACTION_THRESHOLD > 0){ // Can't diverge, compile-time constant
						size_t innerStartOfs;
						for(innerStartOfs = *nGLCt; innerStartOfs >= INTERACTION_THRESHOLD; innerStartOfs -= threadsPerPart){
							ptrdiff_t toGrab = innerStartOfs - threadsPerPart + (threadIdx.x / tgInfo.childCount);
							if(toGrab >= 0){
								interaction = interaction + calc_force(particle.m, particle.pos, nGList[toGrab].mass, nGList[toGrab].barycenter, softening);
							}
						}
						// Need to update stack pointer
						
						for(innerStartOfs = *nGLCt; innerStartOfs >= INTERACTION_THRESHOLD; innerStartOfs -= threadsPerPart){
							ptrdiff_t toGrab = innerStartOfs - threadsPerPart + (threadIdx.x / tgInfo.childCount);
							if(toGrab >= 0){
								interaction = interaction + calc_force(particle.m, particle.pos, pGList[toGrab].m, pGList[toGrab].pos, softening);
							}
						}
						// Need to update stack pointer

						
					}
					
					
					startOfs -= blockDim.x;
				}
				
				swap<Node<DIM, Float>*>(currentLevel, nextLevel);
				swap<size_t*>(cLCt, nLCt);
				curDepth += 1;
			}
			
			// Process remaining interactions and reduce if multithreading in play
		}
	}
	
}


template<size_t DIM, typename Float, size_t PPG, size_t MAX_LEVELS, size_t INTERACTION_THRESHOLD, TraverseMode Mode>
void traverseTreeCUDA(size_t nGroups, GroupInfo<DIM, Float, PPG>* groupInfo, size_t startDepth,
				  Node<DIM, Float>* treeLevels[MAX_LEVELS], size_t treeCounts[MAX_LEVELS], Particle<DIM, Float>* particles, Vec<DIM, Float>* interactions, Float softening, Float theta, size_t blockCt, size_t threadCt){
	
	unsigned char *bfsStackBuffers;
	const size_t stackCapacity = 0;
	
	traverseTreeKernel<DIM, Float, PPG, MAX_LEVELS, INTERACTION_THRESHOLD, Mode><<<blockCt, threadCt>>>(nGroups, groupInfo, startDepth, treeLevels, treeCounts, particles, interactions, softening, theta, bfsStackBuffers, stackCapacity);
	
}

template void traverseTreeCUDA<3, float, 16, 8, 8, Forces>(size_t, GroupInfo<3, float, 16> *, size_t, Node<3, float> **, size_t *, Particle<3, float> *, Vec<3, float> *, float, float, size_t, size_t);


