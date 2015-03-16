#include "treedefs.h"
#include "treecodeCU.h"
#include "cudahelper.h"
#include <iostream>

//: We should really be using native CUDA vectors for this.... but that requires more funny typing magic to convert the CPU data

template<size_t DIM, typename Float, size_t PPG> __device__ bool passesMAC(GroupInfo<DIM, Float, PPG> groupInfo, Node<DIM, Float> nodeHere, Float theta) {
	
	Float d = mag(groupInfo.center - nodeHere.barycenter) - groupInfo.radius;
	Float l = 2 * nodeHere.radius;
	return d > (l / theta);
	
}

template<template<size_t, typename> class StackElms, size_t DIM, typename Float> __device__ void initStack(StackElms<DIM, Float>* level, size_t levelCt, StackElms<DIM, Float>* stack, size_t* stackCt, const size_t capacity){
	//*
	if(threadIdx.x < levelCt){
		stack[threadIdx.x] = level[threadIdx.x];
	}
	//*/
	//*
	if(threadIdx.x == 0){
		*stackCt = levelCt; // We are initializing the stack, so no need to increment previous value
		//printf("%d.%d: %lu = %lu\n", blockIdx.x, threadIdx.x, *stackCt, levelCt);
	}
	//*/
}

template<template<size_t, typename> class StackElms, size_t DIM, typename Float> __device__ void pushAll(const StackElms<DIM, Float>* nodes, const size_t nodeCt, StackElms<DIM, Float>* stack, size_t* stackCt){
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
__global__ void traverseTreeKernel(const size_t nGroups, const GroupInfo<DIM, Float, PPG>* groupInfo,
								   const size_t startDepth, Node<DIM, Float>** treeLevels, const size_t* treeCounts,
								   const Particle<DIM, Float>* particles, const InteractionType<DIM, Float, Mode>* interactions,
								   const Float softening, const Float theta,
								   size_t *bfsStackCounters, Node<DIM, Float> *bfsStackBuffers, const size_t stackCapacity) {
	
	__shared__ size_t interactionCounters[2];
	__shared__ Particle<DIM, Float> particleInteractionList[2*INTERACTION_THRESHOLD];
	__shared__ Node<DIM, Float> nodeInteractionList[2*INTERACTION_THRESHOLD];
	
	if(blockIdx.x >= nGroups) return; // This probably shouldn't happen?
	else {
		GroupInfo<DIM, Float, PPG> tgInfo = groupInfo[blockIdx.x];
		int threadsPerPart = blockDim.x / tgInfo.childCount;
		
		
		size_t* pGLCt = interactionCounters;
		Particle<DIM, Float>* pGList = particleInteractionList;
		initStack((Particle<DIM, Float>*)nullptr, 0, pGList, pGLCt, 2 * INTERACTION_THRESHOLD);
		
		size_t* nGLCt = interactionCounters + 1;
		Node<DIM, Float>* nGList = nodeInteractionList;
		initStack((Node<DIM, Float>*)nullptr, 0, nGList, nGLCt, 2 * INTERACTION_THRESHOLD);
		
		size_t* cLCt = bfsStackCounters + 2 * blockIdx.x;
		Node<DIM, Float>* currentLevel = bfsStackBuffers + 2 * blockIdx.x * stackCapacity;
		
		initStack(treeLevels[startDepth], treeCounts[startDepth], currentLevel, cLCt, stackCapacity);
		
		
		size_t* nLCt = bfsStackCounters + 2 * blockIdx.x + 1;
		Node<DIM, Float>* nextLevel = bfsStackBuffers + (2 * blockIdx.x + 1) * stackCapacity;
		
		__syncthreads();
		
		Particle<DIM, Float> particle;
		if(threadIdx.x > threadsPerPart * tgInfo.childCount){
			particle = particles[tgInfo.childStart + (threadIdx.x % tgInfo.childCount)];
		}
		
		
		InteractionType<DIM, Float, Mode> interaction = freshInteraction<DIM, Float, Mode>();
		size_t curDepth = startDepth;
		while(*cLCt != 0 ){//&& curDepth < MAX_LEVELS){ // Second condition shouldn't matter....
			if(threadIdx.x == 0){
				*nLCt = 0;
			}
			__syncthreads();
			
			ptrdiff_t startOfs = *cLCt;
			while(startOfs > 0){
				ptrdiff_t toGrab = startOfs - blockDim.x + threadIdx.x;
				if(toGrab >= 0){
					Node<DIM, Float> nodeHere = currentLevel[toGrab];
					//*
					if(passesMAC(tgInfo, nodeHere, theta)){
						if(INTERACTION_THRESHOLD > 0){
							// Store to C/G list
							pushAll(&nodeHere, 1, nGList, nGLCt);
						} else if(threadIdx.x < tgInfo.childCount){
							//interaction = interaction + calc_force(particle.m, particle.pos, nodeHere.mass, nodeHere.barycenter, softening);
						}
					} else {
						if(nodeHere.isLeaf){
							if(INTERACTION_THRESHOLD > 0){
								// Store to P/G list
								//pushAll(particles + nodeHere.childStart, nodeHere.childCount, pGList, pGLCt);
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
							//printf("%d.%d: %lu %lu %lu %lu\n", blockIdx.x, threadIdx.x, curDepth, nodeHere.childStart, nodeHere.childCount, *nLCt);
							//if(curDepth + 1 < MAX_LEVELS && nodeHere.childStart < treeCounts[curDepth + 1]){
							pushAll(treeLevels[curDepth + 1] + nodeHere.childStart, nodeHere.childCount, nextLevel, nLCt);
							//}
						}
					}
					//*/
				}
				
				/*
					__syncthreads();
					if(INTERACTION_THRESHOLD > 0){ // Can't diverge, compile-time constant
				 ptrdiff_t innerStartOfs;
				 for(innerStartOfs = *nGLCt; innerStartOfs >= INTERACTION_THRESHOLD; innerStartOfs -= threadsPerPart){
				 ptrdiff_t toGrab = innerStartOfs - threadsPerPart + (threadIdx.x / tgInfo.childCount);
				 if(toGrab >= 0){
				 interaction = interaction + calc_force(particle.m, particle.pos, nGList[toGrab].mass, nGList[toGrab].barycenter, softening);
				 }
				 }
				 // Need to update stack pointer
				 if(threadIdx.x == 0){
				 *nGLCt = (innerStartOfs < 0) ? 0 : innerStartOfs;
				 }
				 
				 for(innerStartOfs = *pGLCt; innerStartOfs >= INTERACTION_THRESHOLD; innerStartOfs -= threadsPerPart){
				 ptrdiff_t toGrab = innerStartOfs - threadsPerPart + (threadIdx.x / tgInfo.childCount);
				 if(toGrab >= 0){
				 interaction = interaction + calc_force(particle.m, particle.pos, pGList[toGrab].m, pGList[toGrab].pos, softening);
				 }
				 }
				 // Need to update stack pointer
				 // Need to update stack pointer
				 if(threadIdx.x == 0){
				 *pGLCt = (innerStartOfs < 0) ? 0 : innerStartOfs;
				 }
				 
				 
					}
					//*/
				
				
				startOfs -= blockDim.x;
			}
			
			swap<Node<DIM, Float>*>(currentLevel, nextLevel);
			size_t oldC = *cLCt;
			size_t oldN = *nLCt;
			swap<size_t*>(cLCt, nLCt);
			//printf("%lu and %lu swapped to %lu and %lu\n",oldC, oldN, *cLCt, *nLCt);
			curDepth += 1;
		}
		
		// Process remaining interactions and reduce if multithreading in play
		
		//*/
		
	}
	
}

template<size_t DIM, typename Float, size_t MAX_LEVELS>
void makeDeviceTree(Node<DIM, Float>* treeLevels[MAX_LEVELS], Node<DIM, Float>* placeHolderTree[MAX_LEVELS], size_t treeCounts[MAX_LEVELS]){
	for(size_t i = 0; i < MAX_LEVELS; i++){
		Node<DIM, Float>* level;
		gpuErrchk( (cudaMalloc(&level, treeCounts[i]*sizeof(Node<DIM, Float>)) )); // We know how big the tree is now. Don't make extra space
		gpuErrchk( (cudaMemcpy(level, treeLevels[i], treeCounts[i]*sizeof(Node<DIM, Float>), cudaMemcpyHostToDevice)) );
		placeHolderTree[i] = level;
	}
}

template<size_t DIM, typename Float, size_t MAX_LEVELS>
void freeDeviceTree(Node<DIM, Float>* placeHolderTree[MAX_LEVELS]){
	for(size_t i = 0; i < MAX_LEVELS; i++){
		gpuErrchk( (cudaFree(placeHolderTree[i])) );
	}
}

// Something is badly wrong with template resolution if we switch to InteractionType here.
// I think the compilers are doing name-mangling differently or something
template<size_t DIM, typename Float, size_t PPG, size_t MAX_LEVELS, size_t INTERACTION_THRESHOLD, TraverseMode Mode>
void traverseTreeCUDA(size_t nGroups, GroupInfo<DIM, Float, PPG>* groupInfo, size_t startDepth,
					  Node<DIM, Float>* treeLevels[MAX_LEVELS], size_t treeCounts[MAX_LEVELS], size_t n, Particle<DIM, Float>* particles, Vec<DIM, Float>* interactions, Float softening, Float theta, size_t blockCt, size_t threadCt){
	
	Node<DIM, Float>* placeHolderLevels[MAX_LEVELS];
	makeDeviceTree<DIM, Float, MAX_LEVELS>(treeLevels, placeHolderLevels, treeCounts);
	Node<DIM, Float>** cuTreeLevels;
	gpuErrchk( (cudaMalloc(&cuTreeLevels, MAX_LEVELS*sizeof(Node<DIM, Float>*))) );
	gpuErrchk( (cudaMemcpy(cuTreeLevels, placeHolderLevels, sizeof(Node<DIM, Float>*), cudaMemcpyHostToDevice)) );
	
	size_t* cuTreeCounts;
	gpuErrchk( (cudaMalloc(&cuTreeCounts, MAX_LEVELS * sizeof(size_t))) );
	gpuErrchk( (cudaMemcpy(cuTreeCounts, treeCounts, MAX_LEVELS * sizeof(size_t), cudaMemcpyHostToDevice)) );
	
	
	size_t biggestRow = 0;
	for(size_t level = 0; level < MAX_LEVELS; level++){
		biggestRow = (treeCounts[level] > biggestRow) ? treeCounts[level] : biggestRow;
	}
	
	const size_t stackCapacity = biggestRow;
	Node<DIM, Float> *bfsStackBuffers;
	size_t * bfsStackCounters;
	gpuErrchk( (cudaMalloc(&bfsStackBuffers, blockCt * 2 * stackCapacity * sizeof(Node<DIM, Float>))) );
	gpuErrchk( (cudaMalloc(&bfsStackCounters, blockCt * 2 * sizeof(size_t))) );
	
	
	GroupInfo<DIM, Float, PPG>* cuGroupInfo;
	gpuErrchk( (cudaMalloc(&cuGroupInfo, nGroups * sizeof(GroupInfo<DIM, Float, PPG>))) );
	gpuErrchk( (cudaMemcpy(cuGroupInfo, groupInfo, nGroups * sizeof(GroupInfo<DIM, Float, PPG>), cudaMemcpyHostToDevice)) );
	
	Particle<DIM, Float>* cuParticles;
	gpuErrchk( (cudaMalloc(&cuParticles, n * sizeof(Particle<DIM, Float>))) );
	gpuErrchk( (cudaMemcpy(cuParticles, particles, n * sizeof(Particle<DIM, Float>), cudaMemcpyHostToDevice)) );
	InteractionType<DIM, Float, Mode>* cuInteractions;
	gpuErrchk( (cudaMalloc(&cuInteractions, n * sizeof(InteractionType<DIM, Float, Mode>))) );
	gpuErrchk( (cudaMemcpy(cuInteractions, interactions, n * sizeof(InteractionType<DIM, Float, Mode>), cudaMemcpyHostToDevice)) );
	
	dim3 dimGrid(blockCt);
	dim3 dimBlock(threadCt);
	std::cout << "Trying to launch with " << threadCt << " / block with " << blockCt << " blocks" << std::endl;
	
	traverseTreeKernel<DIM, Float, PPG, MAX_LEVELS, INTERACTION_THRESHOLD, Mode><<<dimGrid, dimBlock>>>(nGroups, cuGroupInfo, startDepth, cuTreeLevels, cuTreeCounts, cuParticles, cuInteractions, softening, theta, bfsStackCounters, bfsStackBuffers, stackCapacity);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	
	gpuErrchk( (cudaMemcpy(interactions, cuInteractions, n * sizeof(InteractionType<DIM, Float, Mode>), cudaMemcpyDeviceToHost)) );
	
	gpuErrchk( (cudaFree(cuInteractions)) );
	gpuErrchk( (cudaFree(cuParticles)) );
	gpuErrchk( (cudaFree(cuGroupInfo)) );
	gpuErrchk( (cudaFree(bfsStackBuffers)) );
	gpuErrchk( (cudaFree(cuTreeCounts)) );
	freeDeviceTree<DIM, Float, MAX_LEVELS>(placeHolderLevels);
	gpuErrchk( (cudaFree(cuTreeLevels)) );
	
	
	
}

template void traverseTreeCUDA<3, float, 16, 8, 8, Forces>(size_t, GroupInfo<3, float, 16> *, size_t, Node<3, float> **, size_t *, size_t, Particle<3, float> *, InteractionType<3, float, Forces> *, float, float, size_t, size_t);


