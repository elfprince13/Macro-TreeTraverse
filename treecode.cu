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

template<size_t DIM, typename Float> __device__ void initParticleStack(ParticleArray<DIM, Float> level, size_t levelCt, ParticleArray<DIM, Float> stack, size_t* stackCt, const size_t capacity){
	//*
	if(threadIdx.x == 0 && blockIdx.x == 0) printf("%d Copying to stack %lu particles at %p\n",blockIdx.x,levelCt,stack.m);
	if(threadIdx.x < levelCt){
		// This should work with the proxies, with more or less the same code,
		// but doesn't appear to be doing so
		if(threadIdx.x == 0 && blockIdx.x == 0){
			printf("%d Copied particle (%f) to (%f)\n", blockIdx.x, stack.m[threadIdx.x], level.m[threadIdx.x]);
		}
		for(size_t j = 0; j < DIM; j++){
			stack.pos.x[j][threadIdx.x] = level.pos.x[j][threadIdx.x];
		}
		for(size_t j = 0; j < DIM; j++){
			stack.vel.x[j][threadIdx.x] = level.vel.x[j][threadIdx.x];
		}
		stack.m[threadIdx.x] = level.m[threadIdx.x];
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
template<size_t DIM, typename Float> __device__ void initNodeStack(NodeArray<DIM, Float> level, size_t levelCt, NodeArray<DIM, Float> stack, size_t* stackCt, const size_t capacity){
	//*
	if(threadIdx.x == 0 && blockIdx.x == 0) printf("%d Copying to stack %lu nodes at %p\n",blockIdx.x,levelCt,stack.childCount);
	if(threadIdx.x < levelCt){
		// This should work with the proxies, with more or less the same code,
		// but doesn't appear to be doing so
		stack.isLeaf[threadIdx.x] = level.isLeaf[threadIdx.x];
		stack.childCount[threadIdx.x] = level.childCount[threadIdx.x];
		stack.childStart[threadIdx.x] = level.childStart[threadIdx.x];
		if(threadIdx.x == 0 && blockIdx.x == 0){
			printf("%d Copied node (%lu %lu) to (%lu %lu)\n", blockIdx.x, stack.childCount[threadIdx.x], stack.childStart[threadIdx.x], level.childCount[threadIdx.x],level.childStart[threadIdx.x]);
		}
		for(size_t j = 0; j < DIM; j++){
			stack.minX.x[j][threadIdx.x] = level.minX.x[j][threadIdx.x];
		}
		for(size_t j = 0; j < DIM; j++){
			stack.maxX.x[j][threadIdx.x] = level.maxX.x[j][threadIdx.x];
		}
		for(size_t j = 0; j < DIM; j++){
			stack.barycenter.x[j][threadIdx.x] = level.barycenter.x[j][threadIdx.x];
		}
		stack.mass[threadIdx.x] = level.mass[threadIdx.x];
		stack.radius[threadIdx.x] = level.radius[threadIdx.x];
	}
	//*/
	//*
	if(threadIdx.x == 0){
		*stackCt = levelCt; // We are initializing the stack, so no need to increment previous value
							//printf("%d.%d: %lu = %lu\n", blockIdx.x, threadIdx.x, *stackCt, levelCt);
	}
	//*/
}

template<size_t DIM, typename Float> __device__ void pushAllNodes(const NodeArray<DIM, Float> nodes, const size_t nodeCt, NodeArray<DIM, Float> stack, size_t* stackCt){
	if(threadIdx.x == 0 && blockIdx.x == 0) printf("%d Pushing to stack %lu nodes at %p\n",blockIdx.x,nodeCt,stack.childCount);
	
	// This is a weird compiler bug. There's no reason this shouldn't have worked without the cast.
	size_t dst = atomicAdd((unsigned long long*)stackCt, (unsigned long long)nodeCt);
	for(size_t i = dst, j = 0; i < dst+ nodeCt; i++, j++){
		stack.isLeaf[i] = nodes.isLeaf[j];
		stack.childCount[i] = nodes.childCount[j];
		stack.childStart[i] = nodes.childStart[j];
		if(blockIdx.x == 0){
			printf("%d.%d Pushed node (%lu %lu) to (%lu %lu)\n",
				   blockIdx.x,threadIdx.x, stack.childCount[i], stack.childStart[i], nodes.childCount[j],nodes.childStart[j]);
		}
		for(size_t k = 0; k < DIM; k++){
			stack.minX.x[k][i] = nodes.minX.x[k][j];
		}
		for(size_t k = 0; j < DIM; j++){
			stack.maxX.x[k][i] = nodes.maxX.x[k][j];
		}
		for(size_t k = 0; j < DIM; j++){
			stack.barycenter.x[k][i] = nodes.barycenter.x[k][j];
		}
		stack.mass[i] = nodes.mass[j];
		stack.radius[i] = nodes.radius[j];
	}
}

template<size_t DIM, typename Float> __device__ void pushAllParticles(const ParticleArray<DIM, Float> nodes, const size_t nodeCt, ParticleArray<DIM, Float> stack, size_t* stackCt){
	if(threadIdx.x == 0 && blockIdx.x == 0) printf("%d Pushing to stack %lu particles at %p\n",blockIdx.x,nodeCt,stack.m);
	
	// This is a weird compiler bug. There's no reason this shouldn't have worked without the cast.
	size_t dst = atomicAdd((unsigned long long*)stackCt, (unsigned long long)nodeCt);
	for(size_t i = dst, j = 0; i < dst+ nodeCt; i++, j++){
		if(blockIdx.x == 0){
			printf("%d Pushing particle (%f) to (%f)\n", blockIdx.x, stack.m[i], nodes.m[j]);
		}
		for(size_t k = 0; k < DIM; k++){
			stack.pos.x[k][i] = nodes.pos.x[k][j];
		}
		for(size_t k = 0; j < DIM; j++){
			stack.vel.x[k][i] = nodes.vel.x[k][j];
		}
		
		stack.m[i] = nodes.m[j];
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
								   const ParticleArray<DIM, Float> particles, const InteractionTypeArray<DIM, Float, Mode> interactions,
								   const Float softening, const Float theta,
								   size_t *bfsStackCounters, NodeArray<DIM, Float> bfsStackBuffers, const size_t stackCapacity) {
	
	__shared__ size_t interactionCounters[2];
	__shared__ Float particleMass[2*INTERACTION_THRESHOLD];
	__shared__ Float particlePos[DIM * 2*INTERACTION_THRESHOLD];
	__shared__ Float particleVel[DIM * 2*INTERACTION_THRESHOLD];
	
	__shared__ bool nodeLeaves[2 * INTERACTION_THRESHOLD];
	__shared__ size_t nodeChildCount[2*INTERACTION_THRESHOLD];
	__shared__ size_t nodeChildStart[2*INTERACTION_THRESHOLD];
	__shared__ Float nodeMinX[3*2*INTERACTION_THRESHOLD];
	__shared__ Float nodeMaxX[3*2*INTERACTION_THRESHOLD];
	__shared__ Float nodeBarycenter[3*2*INTERACTION_THRESHOLD];
	__shared__ Float nodeMass[2*INTERACTION_THRESHOLD];
	__shared__ Float nodeRadius[2*INTERACTION_THRESHOLD];
	
	
	ParticleArray<DIM, Float> particleInteractionList;
	particleInteractionList.m = particleMass;
	for(size_t j = 0; j < DIM; j++){
		particleInteractionList.pos.x[j] = particlePos + (j * 2 * INTERACTION_THRESHOLD);
	}
	for(size_t j = 0; j < DIM; j++){
		particleInteractionList.vel.x[j] = particleVel + (j * 2 * INTERACTION_THRESHOLD);
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
	
	
	if(blockIdx.x >= nGroups) return; // This probably shouldn't happen?
	else {
		GroupInfo<DIM, Float, PPG> tgInfo = groupInfo[blockIdx.x];
		int threadsPerPart = blockDim.x / tgInfo.childCount;
		
		
		size_t* pGLCt = interactionCounters;
		ParticleArray<DIM, Float> pGList = particleInteractionList;
		ParticleArray<DIM, Float> dummyP;
		initParticleStack(dummyP, 0, pGList, pGLCt, 2 * INTERACTION_THRESHOLD);
		
		
		size_t* nGLCt = interactionCounters + 1;
		NodeArray<DIM, Float> nGList = nodeInteractionList;
		NodeArray<DIM, Float> dummyN;
		initNodeStack(dummyN, 0, nGList, nGLCt, 2 * INTERACTION_THRESHOLD);
		
		size_t* cLCt = bfsStackCounters + 2 * blockIdx.x;
		NodeArray<DIM, Float> currentLevel = bfsStackBuffers + 2 * blockIdx.x * stackCapacity;
		
		if(threadIdx.x == 0 && blockIdx.x == 0){
			printf("%d initing stack: %lu \n", blockIdx.x, startDepth);
			printf("%d continues: %p %p\n",blockIdx.x,treeLevels[startDepth].childCount, treeLevels[startDepth].childStart);
			printf("%d contains: %lu %lu\n",blockIdx.x,treeLevels[startDepth].childCount[0], treeLevels[startDepth].childStart[0]);
			//treeLevels[startDepth].childCount[0], treeLevels[startDepth].childStart[0]
		}
		initNodeStack(treeLevels[startDepth], treeCounts[startDepth], currentLevel, cLCt, stackCapacity);
		
		
		size_t* nLCt = bfsStackCounters + 2 * blockIdx.x + 1;
		NodeArray<DIM, Float> nextLevel = bfsStackBuffers + (2 * blockIdx.x + 1) * stackCapacity;
		
		__syncthreads();
		if(threadIdx.x == 0 && blockIdx.x == 0){
			printf("%d post init stack: %lu \n", blockIdx.x, startDepth);
			printf("%d continues: %p %p\n",blockIdx.x,currentLevel.childCount, currentLevel.childStart);
			printf("%d contains: %lu %lu\n",blockIdx.x,currentLevel.childCount[0], currentLevel.childStart[0]);
			//treeLevels[startDepth].childCount[0], treeLevels[startDepth].childStart[0]
		}
		
		
		
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
			
			__threadfence_block();
			__syncthreads();
			
			ptrdiff_t startOfs = *cLCt;
			while(startOfs > 0){
				ptrdiff_t toGrab = startOfs - blockDim.x + threadIdx.x;
				if(toGrab >= 0){
					Node<DIM, Float> nodeHere = currentLevel[toGrab];
					if(threadIdx.x == 0 &&
					   blockIdx.x == 0){
						printf("%d.%d @ %lu:\t%lu %lu vs %lu %lu with %lu %ld \n", blockIdx.x, threadIdx.x, curDepth, nodeHere.childStart, nodeHere.childCount, currentLevel.childStart[toGrab], currentLevel.childCount[toGrab], *cLCt, toGrab);
					}
					//*
					if(passesMAC(tgInfo, nodeHere, theta)){
						if(INTERACTION_THRESHOLD > 0){
							// Store to C/G list
							pushAllNodes(NodeArray<DIM, Float>(nodeHere), 1, nGList, nGLCt);
						} else if(threadIdx.x < tgInfo.childCount){
							//interaction = interaction + calc_force(particle.m, particle.pos, nodeHere.mass, nodeHere.barycenter, softening);
						}
					} else {
						if(nodeHere.isLeaf){
							if(INTERACTION_THRESHOLD > 0){
								// Store to P/G list
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
							pushAllNodes(treeLevels[curDepth + 1] + nodeHere.childStart, nodeHere.childCount, nextLevel, nLCt);
							//}
						}
					}
					//*/
				}
				__threadfence_block();
				__syncthreads();
				//*
				if(INTERACTION_THRESHOLD > 0){ // Can't diverge, compile-time constant
					ptrdiff_t innerStartOfs;
					for(innerStartOfs = *nGLCt; innerStartOfs >= INTERACTION_THRESHOLD; innerStartOfs -= threadsPerPart){
						ptrdiff_t toGrab = innerStartOfs - threadsPerPart + (threadIdx.x / tgInfo.childCount);
						if(toGrab >= 0){
							Vec<DIM, Float> bc = nGList[toGrab].barycenter;
							interaction = interaction + calc_force(particle.m, particle.pos, *nGList[toGrab].mass, bc, softening);
						}
					}
					// Need to update stack pointer
					if(threadIdx.x == 0){
						*nGLCt = (innerStartOfs < 0) ? 0 : innerStartOfs;
					}
				 
					for(innerStartOfs = *pGLCt; innerStartOfs >= INTERACTION_THRESHOLD; innerStartOfs -= threadsPerPart){
						ptrdiff_t toGrab = innerStartOfs - threadsPerPart + (threadIdx.x / tgInfo.childCount);
						if(toGrab >= 0){
							Vec<DIM, Float> c = pGList[toGrab].pos;
							interaction = interaction + calc_force(particle.m, particle.pos, *pGList[toGrab].m, c, softening);
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
			
			if(threadIdx.x == 0 && blockIdx.x == 0)printf("Done inside\n");
			
			swap<NodeArray<DIM, Float>>(currentLevel, nextLevel);
			swap<size_t*>(cLCt, nLCt);
			//printf("%lu and %lu swapped to %lu and %lu\n",oldC, oldN, *cLCt, *nLCt);
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
	
	traverseTreeKernel<DIM, Float, PPG, MAX_LEVELS, INTERACTION_THRESHOLD, Mode><<<dimGrid, dimBlock>>>(nGroups, cuGroupInfo, startDepth, cuTreeLevels, cuTreeCounts, cuParticles, cuInteractions, softening, theta, bfsStackCounters, bfsStackBuffers, stackCapacity);
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


