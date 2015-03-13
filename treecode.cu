#include <treedefs.h>

template<size_t DIM, typename Float, size_t PPG> __device__ bool passesMAC(GroupInfo<DIM, Float, PPG> groupInfo, Node<DIM, Float> nodeHere, Float theta) {
	
	Float d = mag(groupInfo.center, nodeHere.barycenter) - groupInfo.radius;
	Float l = 2 * nodeHere.radius;
	return d > (l / theta);
	
}


template<size_t DIM, typename Float, size_t PPG> __global__ void traverseTree(int nGroups, GroupInfo<DIM, Float, PPG>* groupInfo, int startDepth, Node<DIM>*[MaxLevels] treeLevels, Particle<DIM, Float>* particles, Vec<DIM, Float>* interactions) {
	if(blockIdx.x >= nGroups) return;
	else {
		GroupInfo tgInfo = groupInfo[blockIdx.x];
		int threadsPerPart = blockDim.x / tgInfo.nParts;
		if(threadIdx.x > threadsPerPart * tgInfo.nParts) return;
		else {
			NodeStack currentLevel = initNodeStack(treeLevels[curDepth]); // Shared?
			NodeStack nextLevel = emptyNodeStack(); // Shared?
			
			Particle particle = particles[partIds[threadIdx.x % tgInfo.nParts]];
			Interaction interaction = freshInteraction(dims);
			int curDepth = startDepth;
			while(currentLevel.size){
				nextLevel.clear(); // Only once please
				__syncthreads();
			
				int startOfs = 0;
				while(startOfs < currentLevel.size){
					Node nodeHere = currentLevel[startOfs + threadIdx.x];
					if(passesMAC(tgInfo, nodeHere)){
						// Store to C/G list
					} else {
						if(isLeaf(nodeHere)){
							// Store to P/G list
						} else {
							push(nodeHere.children);
						}
					}
					__syncthreads();
					
					int innerStartOfs = 0;
					// Interact - SOMETHING IS FISHY HERE
					// Use a 2 * blockDims.x circular buffer for CGList?
					while(innerStartOfs + blocksDims.x <= CGList.size){
						interaction += calcCellInteraction(particle, CGList[this half])
						
						innerStartOfs += blockDims.x
					}
					
					innerStartOfs = 0;
					// Interact
					while(innerStartOfs + blocksDims.x <= PGList.size){
						interaction += calcPartInteraction(particle, PGList[innerStartOfs + threadIdx.x])
						
						innerStartOfs += blockDims.x
					}
					
					startOfs += blockDims.x
				}
				
				swap(currentLevel, nextLevel);
				curDepth += 1;
			}			
		}
	}
	
}

