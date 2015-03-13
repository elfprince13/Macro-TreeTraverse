#include "treedefs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>

//: We should really be using native CUDA vectors for this.... but that requires more funny typing magic to convert the CPU data

template<size_t DIM, typename Float, size_t PPG> __device__ bool passesMAC(GroupInfo<DIM, Float, PPG> groupInfo, Node<DIM, Float> nodeHere, Float theta) {
	
	Float d = mag(groupInfo.center, nodeHere.barycenter) - groupInfo.radius;
	Float l = 2 * nodeHere.radius;
	return d > (l / theta);
	
}

template<size_t DIM, typename Float> __device__ void initNodeStack(Node<DIM, Float>* level, size_t levelCt, Node<DIM, Float>* stack, size_t* stackCt){
	if(threadIdx.x < levelCt){
		stack[threadIdx.x] = level[threadIdx.x];
		if(threadIdx.x == 0){
			*stackCt += levelCt;
		}
	}
}

template<size_t DIM, typename Float> __device__ void pushAll(Node<DIM, Float>* nodes, int nodeCt, Node<DIM, Float>* stack, int* stackCt){
	int dst = atomicAdd(stackCt, nodeCt);
	for(int i = dst, j = 0; i < dst+ nodeCt; i++, j++){
		stack[i] = nodes[j];
	}
}

// Needs softening
template<size_t DIM, typename Float> __device__ Vec<DIM, Float> calc_force(Float m1, Vec<DIM, Float> v1, Float m2, Vec<DIM, Float> v2, Float softening){
	Vec<DIM, Float> disp = v1 - v2;
	Vec<DIM, Float> force;
	force = disp * ((*m1 * m2) / (Float)(softening + pow(mag_sq(disp),1.5)));
	return force;
}

template<size_t DIM, typename Float> __device__ Vec<DIM, Float> freshInteraction(){
	Vec<DIM, Float> fresh; for(size_t i = 0; i < DIM; i++){
		fresh[i] = 0.0;
	}
	return fresh;
}

template<typename T> __device__ inline void swap(T& a, T& b){
	T c(a); a=b; b=c;
}

template<size_t DIM, typename Float, size_t PPG, size_t MAX_LEVELS>
__global__ void traverseTree(size_t nGroups, GroupInfo<DIM, Float, PPG>* groupInfo, size_t startDepth,
							 Node<DIM, Float>* treeLevels[MAX_LEVELS], size_t treeCounts[MAX_LEVELS], Particle<DIM, Float>* particles, Vec<DIM, Float>* interactions) {
	extern __shared__ unsigned char smem;
	
	if(blockIdx.x >= nGroups) return; // This probably shouldn't happen?
	else {
		GroupInfo<DIM, Float, PPG> tgInfo = groupInfo[blockIdx.x];
		int threadsPerPart = blockDim.x / tgInfo.nParts;
		if(threadIdx.x > threadsPerPart * tgInfo.nParts) return;
		else {
			// This is a horrible abuse of shared memory. We need to SOA the stacks
			// Also this is too big
			size_t* cLCt = (size_t*)smem;
			Node<DIM, Float>* currentLevel = (Node<DIM, Float>*)(smem + sizeof(size_t));
			size_t* nLCt = (size_t*)(smem + sizeof(size_t) + 2 * blockDim.x * sizeof(Node<DIM, Float>));
			Node<DIM, Float>* nextLevel = (Node<DIM, Float>*)(smem + 2*sizeof(size_t) +  2 * blockDim.x * sizeof(Node<DIM, Float>));
			if(threadIdx.x == 0){
				*cLCt = 0;
			}
			initNodeStack(treeLevels[startDepth], treeCounts[startDepth], currentLevel, cLCt);
			__syncthreads();
			
			Particle<DIM, Float> particle = particles[tgInfo.childStart + (threadIdx.x % tgInfo.nParts)];
			Vec<DIM, Float> interaction = freshInteraction<DIM, Float>();
			int curDepth = startDepth;
			while(*cLCt != 0){
				if(threadIdx.x == 0){
					*nLCt = 0;
				}
				__syncthreads();
			
				int startOfs = *cLCt;
				while(startOfs > 0){
					int toGrab = startOfs - blockDim.x + threadIdx.x;
					if(toGrab >= 0){
						Node<DIM, Float> nodeHere = currentLevel[toGrab];
						if(passesMAC(tgInfo, nodeHere)){
							// Store to C/G list
						} else {
							if(nodeHere.isLeaf){
								// Store to P/G list
							} else {
								pushAll(treeLevels[curDepth + 1] + nodeHere.childStart, nodeHere.childCount, nextLevel, *nLCt);
							}
						}
					}
					__syncthreads();
					
					int innerStartOfs = 0;
					// Consider transposing either of these loops.
					
					// Interact - SOMETHING IS FISHY HERE
					// Use a 2 * blockDim.x circular buffer for CGList?
					/*
					while(innerStartOfs + blockDim.x <= CGList.size){
						interaction = interaction + calcCellInteraction(particle, CGList[innerStartOfs + threadIdx.x])
						
						innerStartOfs += blockDim.x
					}
					
					innerStartOfs = 0;
					// Interact
					while(innerStartOfs + blockDim.x <= PGList.size){
						interaction = interaction + calc_force(particle.m, particle.pos, PGList[innerStartOfs + threadIdx.x])
						
						innerStartOfs += blockDim.x
					}
					 */
					
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

