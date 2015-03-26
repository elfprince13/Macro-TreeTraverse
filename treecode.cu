#include "treedefs.h"
#include "CudaArrayCopyUtils.h"
#include "treecodeCU.h"
#include "cudahelper.h"
#include <iostream>

//: We should really be using native CUDA vectors for this.... but that requires more funny typing magic to convert the CPU data

template<size_t DIM, typename Float, size_t PPG> __device__ bool passesMAC(const GroupInfo<DIM, Float, PPG>& groupInfo, const Node<DIM, Float>& nodeHere, Float theta) {
	
	Float d = mag(groupInfo.center - nodeHere.barycenter.pos) - groupInfo.radius;
	Float l = 2 * nodeHere.radius;
	return d > (l / theta);
	
}


template<template<size_t, typename> class ElemType, template<size_t, typename> class ElemTypeArray, size_t DIM, typename Float>
__device__ void initStack(ElemTypeArray<DIM, Float> level, size_t levelCt, ElemTypeArray<DIM, Float> stack, size_t* stackCt){
	if(threadIdx.x < levelCt){
		if(levelCt > level.elems || threadIdx.x >= stack.elems) printf("%d.%d %s: %lu (should be <= %lu) elements. stackCt @ %d / %lu\n",blockIdx.x, threadIdx.x, __func__,levelCt,level.elems,threadIdx.x,stack.elems);

		ElemType<DIM, Float> eHere;
		level.get(threadIdx.x, eHere);
		stack.set(threadIdx.x, eHere);
	}
	if(threadIdx.x == 0){
		atomicExch((unsigned long long*)stackCt,(unsigned long long)levelCt); // Don't want to have to threadfence afterwards. Just make sure it's set!
	}
}

template<size_t DIM, typename Float> __device__ void dumpStackChildren(NodeArray<DIM, Float> stack, const size_t* stackCt){
	size_t dst = *stackCt;
	for(size_t i = 0; i < dst; i++){
			printf("(%lu) see (%lu %lu) in (%p/%lu)\n",i,stack.childStart[i],stack.childCount[i], stackCt, dst);
	}
}

template<size_t DIM, typename T> __device__ void pushMeta(const size_t srcSt, const size_t srcCt, PointMassArray<DIM, T> stack, size_t* stackCt){
	// This is a weird compiler bug. There's no reason this shouldn't have worked without the cast.
	size_t dst = atomicAdd((unsigned long long*)stackCt, (unsigned long long)srcCt);
	for(size_t i = dst, j = 0; i < dst + srcCt; i++, j++){
		PointMass<DIM, T> eHere;
		eHere.m = 1;
		eHere.pos.x[0] = srcSt + j;
		eHere.pos.x[1] = 0;
		eHere.pos.x[2] = 0;
		stack.set(i, eHere);
	}
}

template<template<size_t, typename> class ElemType, template<size_t, typename> class ElemTypeArray, size_t DIM, typename Float> __device__ void pushAll(const ElemTypeArray<DIM, Float> src, const size_t srcCt, ElemTypeArray<DIM, Float> stack, size_t* stackCt){
	// This is a weird compiler bug. There's no reason this shouldn't have worked without the cast.
	size_t dst = atomicAdd((unsigned long long*)stackCt, (unsigned long long)srcCt);
	if(srcCt > src.elems || dst >= stack.elems)  printf("%d.%d %s: %lu (should be <= %lu) elements. stackCt @ %lu / %lu\n",blockIdx.x, threadIdx.x, __func__, srcCt,src.elems,dst,stack.elems);
	for(size_t i = dst, j = 0; i < dst + srcCt; i++, j++){
		ElemType<DIM, Float> eHere;
		src.get(j, eHere);
		stack.set(i, eHere);
	}
}

template<size_t DIM, typename Float, TraverseMode Mode> __device__ InteractionType(DIM, Float, Mode) freshInteraction(){
	InteractionType(DIM, Float, Mode) fresh; for(size_t i = 0; i < DIM; i++){
		fresh.x[i] = 0.0;
	}
	return fresh;
}

// Needs softening

template<size_t DIM, typename Float, TraverseMode Mode> __device__ InteractionType(DIM, Float, Mode) calc_interaction(const PointMass<DIM, Float> &m1, const InteracterType(DIM, Float, Mode) &m2, Float softening){
	InteractionType(DIM, Float, Mode) interaction = freshInteraction<DIM, Float, Mode>();
	size_t isParticle = m2.m != 0;
	switch(Mode){
	case Forces:{
		// Reinterpret cast is evil, but necessary here. template-induced dead-code and all.
		const PointMass<DIM, Float>& m2_inner = reinterpret_cast<const PointMass<DIM, Float>& >(m2);
		Vec<DIM, Float> disp = m1.pos - m2_inner.pos;
		interaction = (disp * ((m1.m * m2_inner.m) / (Float)(softening + pow((Float)mag_sq(disp),(Float)1.5))));
		break;}
	case CountOnly:
		interaction.x[isParticle] = 1;
		break;
	case HashInteractions:{
		// Type inference is failing here unless these are all explicitly separate variables
		size_t generic = m2.pos.x[0];
		size_t nodeSpecific1 = m2.pos.x[1];
		size_t nodeSpecific2 = m2.pos.x[2];
		size_t nodeSpecific = nodeSpecific1 ^ nodeSpecific2;
		size_t e = generic ^ (!isParticle) * nodeSpecific;
		interaction.x[isParticle] = e;
		break;}
	}
	return interaction;
}

template<typename T> __device__ inline void swap(T& a, T& b){
	T c(a); a=b; b=c;
}

template<size_t DIM, typename Float, size_t TPB, size_t PPG, size_t MAX_LEVELS, size_t INTERACTION_THRESHOLD, TraverseMode Mode>
__global__ void traverseTreeKernel(const size_t nGroups, const GroupInfoArray<DIM, Float, PPG> groupInfo,
								   const size_t startDepth, NodeArray<DIM, Float>* treeLevels, const size_t* treeCounts,
								   const size_t n, const ParticleArray<DIM, Float> particles, InteractionTypeArray(DIM, Float, Mode) interactions,
								   const Float softening, const Float theta,
								   size_t *bfsStackCounters, NodeArray<DIM, Float> bfsStackBuffers, const size_t stackCapacity) {
typedef typename std::conditional<NonForceCondition(Mode), size_t, Float>::type InteractionElemType;
#define BUF_MUL 128
	// TPB must = blockDims.x
	__shared__ size_t interactionCounters[1];
	__shared__ InteractionElemType pointMass[BUF_MUL*INTERACTION_THRESHOLD];
	__shared__ InteractionElemType pointPos[DIM * BUF_MUL*INTERACTION_THRESHOLD];
	__shared__ InteractionElemType interactionBuf[TPB * InteractionElems(Mode, DIM, 2)];
	
	if(threadIdx.x == 0 && blockDim.x != TPB){
		printf("%d Launched with mismatched TPB parameters\n",blockIdx.x);
	}

	InteracterTypeArray(DIM, Float, Mode) interactionList;
	interactionList.m = pointMass;
	for(size_t j = 0; j < DIM; j++){
		interactionList.pos.x[j] = pointPos + (j * BUF_MUL*INTERACTION_THRESHOLD);
	}
	interactionList.setCapacity(BUF_MUL*INTERACTION_THRESHOLD);
	
	InteractionTypeArray(DIM, Float, Mode) interactionScratch;
	for(size_t j = 0; j < InteractionElems(Mode, DIM, 2); j++){
		interactionScratch.x[j] = interactionBuf + (TPB * j);
	}
	interactionScratch.setCapacity(TPB);

	if(threadIdx.x == 0 && blockIdx.x == 0){
		printf("%p\n",interactionList.m);
		for(size_t j = 0; j < DIM; j++){
			printf("%p ", interactionList.pos.x[j]);
		} printf("\n");
	}
	//if(threadIdx.x == 0)printf("%3d checking in (0)\n",blockIdx.x);
	
	for(size_t groupOffset = 0; groupOffset + blockIdx.x < nGroups; groupOffset += gridDim.x){
		if(blockIdx.x == 0 && threadIdx.x == 0) printf("%3d checking in with offset %lu / %lu, inc by %d\n",blockIdx.x,groupOffset,nGroups,gridDim.x);
		GroupInfo<DIM, Float, PPG> tgInfo;
		groupInfo.get(blockIdx.x + groupOffset,tgInfo);
		size_t threadsPerPart = blockDim.x / tgInfo.childCount;
		
		
		size_t* pGLCt = interactionCounters;
		InteracterTypeArray(DIM, Float, Mode) pGList = interactionList;
		InteracterTypeArray(DIM, Float, Mode) dummyP;
		initStack<PointMass,PointMassArray>(dummyP, 0, pGList, pGLCt);
		
		size_t* cLCt = bfsStackCounters + 2 * blockIdx.x;
		NodeArray<DIM, Float> currentLevel = bfsStackBuffers + 2 * blockIdx.x * stackCapacity;
		currentLevel.setCapacity(stackCapacity);
		initStack<Node,NodeArray>(treeLevels[startDepth], treeCounts[startDepth], currentLevel, cLCt);
		
		
		size_t* nLCt = bfsStackCounters + 2 * blockIdx.x + 1;
		NodeArray<DIM, Float> nextLevel = bfsStackBuffers + (2 * blockIdx.x + 1) * stackCapacity;
		nextLevel.setCapacity(stackCapacity);

		__threadfence_block();
		__syncthreads(); // Everyone needs the stack initialized before we can continue
		
		
		
		const size_t useful_thread_ct =  threadsPerPart * tgInfo.childCount;
		Particle<DIM, Float> particle;
		if(threadIdx.x < useful_thread_ct){
			if(tgInfo.childStart + (threadIdx.x % tgInfo.childCount) >= particles.elems){
				printf("Getting particle, %d < %lu, so want at %lu + (%d %% %lu) = %lu\n",threadIdx.x,useful_thread_ct,tgInfo.childStart, threadIdx.x, tgInfo.childCount, tgInfo.childStart + (threadIdx.x % tgInfo.childCount));
			}
			particles.get(tgInfo.childStart + (threadIdx.x % tgInfo.childCount), particle);
		}
		
		InteractionType(DIM, Float, Mode) interaction = freshInteraction<DIM, Float, Mode>();
		size_t curDepth = startDepth;
		while(*cLCt != 0 ){
			if(threadIdx.x == 0){
				*nLCt = 0;
			}
			
			__threadfence_block();
			__syncthreads();
			
			ptrdiff_t startOfs = *cLCt;
			while(startOfs > 0){
				ptrdiff_t toGrab = startOfs - blockDim.x + threadIdx.x;
				if(toGrab >= 0){
					Node<DIM, Float> nodeHere;
					currentLevel.get(toGrab, nodeHere);
					//if(threadIdx.x == 0) printf("\t%d.%d @ %lu:\t%lu %lu vs %lu %lu with %lu %ld \n", blockIdx.x, threadIdx.x, curDepth, nodeHere.childStart, nodeHere.childCount, currentLevel.childStart[toGrab], currentLevel.childCount[toGrab], *cLCt, toGrab);
					//*
					if(passesMAC(tgInfo, nodeHere, theta)){
						//if(threadIdx.x == 0) printf("\t%d accepted MAC\n",threadIdx.x);
						if(INTERACTION_THRESHOLD > 0){
							// Store to C/G list
							//if(threadIdx.x == 0) printf("\t%d found the following:\n\t(%lu %lu) @ (%p/%lu), writing to stack at pos %lu @ %p\n", threadIdx.x, nodeHere.childStart, nodeHere.childCount, treeLevels[curDepth + 1].childStart, curDepth + 1, *nLCt,nLCt);
							
							InteracterType(DIM, Float, Mode) nodePush;
							switch(Mode){
							case Forces:{
									nodePush = nodeHere.barycenter; break;}
							case CountOnly:
							case HashInteractions:{
									nodePush.m = 0;
									nodePush.pos.x[0] = curDepth;
									nodePush.pos.x[1] = nodeHere.childStart;
									nodePush.pos.x[2] = nodeHere.childCount;
									break;}
							}
							InteracterTypeArray(DIM, Float, Mode) tmpArray(nodePush);
							size_t tmpCt = 1;
							
							pushAll<PointMass,PointMassArray>(tmpArray, tmpCt, pGList, pGLCt);
						} else if(threadIdx.x < tgInfo.childCount){
							//interaction = interaction + calc_force(particle.m, particle.pos, nodeHere.mass, nodeHere.barycenter, softening);
						}
					} else {
						//if(threadIdx.x == 0) printf("\t%d rejected MAC\n",threadIdx.x);
						if(nodeHere.isLeaf){
							if(INTERACTION_THRESHOLD > 0){
								// Store to P/G list
								//printf("Pushing particles %lu particles, ")
								if(nodeHere.childCount > 16){
									printf("\t%d.%d: Adding a lot particles %lu\n",blockIdx.x,threadIdx.x,nodeHere.childCount);
								}
								switch(Mode){
								case Forces:{
									pushAll<PointMass, PointMassArray>(particles.mass + nodeHere.childStart, nodeHere.childCount, *reinterpret_cast<PointMassArray<DIM, Float>* >(&pGList), pGLCt); break;}
								case CountOnly:
								case HashInteractions:{
									pushMeta(nodeHere.childStart, nodeHere.childCount, pGList, pGLCt);
									break;}
								}

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
							pushAll<Node, NodeArray>(treeLevels[curDepth + 1] + nodeHere.childStart, nodeHere.childCount, nextLevel, nLCt);
						}
					}
					//*/
				}
				__threadfence_block();
				__syncthreads();
				// if(threadIdx.x == 0) printf("\t%3d.%d All safely past toGrab\n",blockIdx.x,threadIdx.x);
				
				//*
				if(INTERACTION_THRESHOLD > 0){ // Can't diverge, compile-time constant
					ptrdiff_t innerStartOfs;
					//if(threadIdx.x == 0) printf("\t%d PGLCt is %lu >? %lu (%ld > %ld)\n",threadIdx.x,*pGLCt,INTERACTION_THRESHOLD,(ptrdiff_t)(*pGLCt),(ptrdiff_t)INTERACTION_THRESHOLD);

					// The casting here feels very strange - why is innerStartOfs implicitly casted, rather than vice versa?
					for(innerStartOfs = *pGLCt; innerStartOfs >= (ptrdiff_t)INTERACTION_THRESHOLD; innerStartOfs -= threadsPerPart){
						ptrdiff_t toGrab = innerStartOfs - threadsPerPart + (threadIdx.x / tgInfo.childCount);
						// printf("\t%d interacting with %ld = %lu - %lu + (%d / %d)\n",threadIdx.x,toGrab,innerStartOfs,threadsPerPart,threadIdx.x,tgInfo.childCount);
						if(toGrab >= 0){
							InteracterType(DIM, Float, Mode) pHere;
							pGList.get(toGrab, pHere);
							interaction = interaction + calc_interaction<DIM, Float, Mode>(particle.mass, pHere, softening);
						}
					}
					//if(threadIdx.x == 0) printf("\t%d through interaction loop safely\n",threadIdx.x);
					// Need to update stack pointer
					if(threadIdx.x == 0){
						atomicExch((unsigned long long *)pGLCt, (unsigned long long)((innerStartOfs < 0) ? 0 : innerStartOfs));
					}
				}
				//*/

				//if(threadIdx.x == 0) printf("%3d.%d: Try going around again\n",blockIdx.x,threadIdx.x);
				startOfs -= blockDim.x;
			}
			
			//if(threadIdx.x == 0) printf("%3d.%d Done inside: %lu (loopcount at %ld) work remaining at depth: %lu\n",blockIdx.x, threadIdx.x, *nLCt,startOfs,curDepth);
			swap<NodeArray<DIM, Float>>(currentLevel, nextLevel);
			swap<size_t*>(cLCt, nLCt);
			curDepth += 1;
		}
		
		// Process remaining interactions
		//*
		__threadfence_block();
		__syncthreads();

		if(INTERACTION_THRESHOLD > 0){ // Can't diverge, compile-time constant
			ptrdiff_t innerStartOfs;

			for(innerStartOfs = *pGLCt; innerStartOfs > 0; innerStartOfs -= threadsPerPart){
				ptrdiff_t toGrab = innerStartOfs - threadsPerPart + (threadIdx.x / tgInfo.childCount);
				if(toGrab >= 0){
					InteracterType(DIM, Float, Mode) pHere;
					pGList.get(toGrab, pHere);
					interaction = interaction + calc_interaction<DIM, Float, Mode>(particle.mass, pHere, softening);
				}
			}
			// Need to update stack pointer
			// Need to update stack pointer
			if(threadIdx.x == 0){
				atomicExch((unsigned long long *)pGLCt, 0);
			}
		}

		// This needs to be done in shared memory! We should figure out how to combine with the stack scratch-space!
		if(threadIdx.x < useful_thread_ct){
			interactionScratch.set(threadIdx.x, interaction);
		}
		//printf("Remainder processed\n");

		__threadfence_block();
		__syncthreads(); // All forces have been summed and are in view

		// reduce (hack-job fashion for now) if multithreading per particle in play
		//*
		//printf("Reducing\n");
		if(threadIdx.x < tgInfo.childCount){
			InteractionType(DIM, Float, Mode) accInt = freshInteraction<DIM, Float, Mode>();
			for(size_t i = 1; i < threadsPerPart; i++){
				InteractionType(DIM, Float, Mode) tmp;
				interactionScratch.get(threadIdx.x + i * tgInfo.childCount, tmp);
				accInt = accInt + tmp;
			}
			interactions.set(tgInfo.childStart + threadIdx.x, interaction + accInt);

		}
		//if(threadIdx.x == 0) printf("%3d Done reducing\n",blockIdx.x);
		//*/
		
	}
	return;
	
}


// Something is badly wrong with template resolution if we switch to InteractionType here.
// I think the compilers are doing name-mangling differently or something


/*
 template void traverseTreeCUDA<3, float, 128, 16, 16, 300000, 8, Forces>
 	 	 	 	 	 (size_t, GroupInfoArray<3, float, 16>, size_t,
 	 	 	 	 	 NodeArray<3, float> *, size_t *,
 	 	 	 	 	 size_t, ParticleArray<3, float>, VecArray<3, float>, float, float, size_t);
*/
template<size_t DIM, typename Float, size_t threadCt, size_t PPG, size_t MAX_LEVELS, size_t MAX_STACK_ENTRIES, size_t INTERACTION_THRESHOLD, TraverseMode Mode>
void traverseTreeCUDA(size_t nGroups, GroupInfoArray<DIM, Float, PPG> groupInfo, size_t startDepth,
					  NodeArray<DIM, Float> treeLevels[MAX_LEVELS], size_t treeCounts[MAX_LEVELS],
					  size_t n, ParticleArray<DIM, Float> particles, InteractionTypeArray(DIM, Float, Mode) interactions, Float softening, Float theta, size_t blockCt){

	std::cout << "Traverse tree with " << blockCt << " blocks and " << threadCt << " tpb"<<std::endl;
	NodeArray<DIM, Float> placeHolderLevels[MAX_LEVELS];
	makeDeviceTree<DIM, Float, MAX_LEVELS>(treeLevels, placeHolderLevels, treeCounts);
	NodeArray<DIM, Float>* cuTreeLevels;

	ALLOC_DEBUG_MSG(MAX_LEVELS*sizeof(NodeArray<DIM, Float>) + MAX_LEVELS * sizeof(size_t));

	gpuErrchk( (cudaMalloc(&cuTreeLevels, MAX_LEVELS*sizeof(NodeArray<DIM, Float>))) );
	gpuErrchk( (cudaMemcpy(cuTreeLevels, placeHolderLevels, MAX_LEVELS*sizeof(NodeArray<DIM, Float>), cudaMemcpyHostToDevice)) );
	
	size_t* cuTreeCounts;
	gpuErrchk( (cudaMalloc(&cuTreeCounts, MAX_LEVELS * sizeof(size_t))) );
	gpuErrchk( (cudaMemcpy(cuTreeCounts, treeCounts, MAX_LEVELS * sizeof(size_t), cudaMemcpyHostToDevice)) );
	
	
	size_t biggestRow = 0;
	for(size_t level = 0; level < MAX_LEVELS; level++){
		biggestRow = (treeCounts[level] > biggestRow) ? treeCounts[level] : biggestRow;
	}


	std::cout << "Biggest row: " << biggestRow  << std::endl;


	const size_t stackCapacity = biggestRow;
	const size_t blocksPerLaunch = MAX_STACK_ENTRIES / stackCapacity;
	std::cout << "Allowing: " << blocksPerLaunch << " blocks per launch" << std::endl;

	NodeArray<DIM, Float> bfsStackBuffers;
	size_t * bfsStackCounters;
	allocDeviceNodeArray(blocksPerLaunch * 2 * stackCapacity, bfsStackBuffers);

	ALLOC_DEBUG_MSG(blocksPerLaunch * 2 * sizeof(size_t));
	gpuErrchk( (cudaMalloc(&bfsStackCounters, blocksPerLaunch * 2 * sizeof(size_t))) );
	
	
	GroupInfoArray<DIM, Float, PPG> cuGroupInfo;
	allocDeviceGroupInfoArray(nGroups, cuGroupInfo);
	copyDeviceGroupInfoArray(nGroups, cuGroupInfo, groupInfo, cudaMemcpyHostToDevice);
	
	ParticleArray<DIM, Float> cuParticles;
	allocDeviceParticleArray(n, cuParticles);
	copyDeviceParticleArray(n, cuParticles, particles, cudaMemcpyHostToDevice);
	
	InteractionTypeArray(DIM, Float, Mode) cuInteractions;
	allocDeviceVecArray(n, cuInteractions);
	copyDeviceVecArray(n, cuInteractions, interactions, cudaMemcpyHostToDevice);
	
	dim3 dimGrid(blocksPerLaunch);
	dim3 dimBlock(threadCt);
	std::cout << "Trying to launch with " << threadCt << " / block with " << blocksPerLaunch << " blocks" << std::endl;
	
	traverseTreeKernel<DIM, Float, threadCt, PPG, MAX_LEVELS, INTERACTION_THRESHOLD, Mode><<<dimGrid, dimBlock>>>(nGroups, cuGroupInfo, startDepth, cuTreeLevels, cuTreeCounts, n, cuParticles, cuInteractions, softening, theta, bfsStackCounters, bfsStackBuffers, stackCapacity);
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


template void traverseTreeCUDA<3, float, 128, 16, 16, 300000, 8, Forces>(size_t, GroupInfoArray<3, float, 16>, size_t, NodeArray<3, float> *, size_t *, size_t, ParticleArray<3, float>, InteractionTypeArray(3, float, Forces), float, float, size_t);
template void traverseTreeCUDA<3, float, 128, 16, 16, 300000, 8, CountOnly>(size_t, GroupInfoArray<3, float, 16>, size_t, NodeArray<3, float> *, size_t *, size_t, ParticleArray<3, float>, InteractionTypeArray(3, float, CountOnly), float, float, size_t);
template void traverseTreeCUDA<3, float, 128, 16, 16, 300000, 8, HashInteractions>(size_t, GroupInfoArray<3, float, 16>, size_t, NodeArray<3, float> *, size_t *, size_t, ParticleArray<3, float>, InteractionTypeArray(3, float, HashInteractions), float, float, size_t);

