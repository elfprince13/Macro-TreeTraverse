//
//  CudaArrayCopyUtils.h
//  TreeTraverse
//
//  Created by Thomas Dickerson on 3/16/15.
//  Copyright (c) 2015 StickFigure Graphic Productions. All rights reserved.
//

#ifndef TreeTraverse_CudaArrayCopyUtils_h
#define TreeTraverse_CudaArrayCopyUtils_h

#include "treedefs.h"
#include "cudahelper.h"
#include <iostream>

#define ALLOC_DEBUG_MSG(N) 	//printf("%s @ %s:%d: Allocating " SZSTR " bytes\n",__func__,__FILE__,__LINE__,N)

//---------------------------------------------------



template<our_size_t DIM, typename Float> void allocDeviceVecArray(our_size_t width, VecArray<DIM, Float>& array){
	our_size_t floatBytes = width*sizeof(Float);
	ALLOC_DEBUG_MSG(floatBytes*DIM);
	for(our_size_t j = 0; j < DIM; j++){
		gpuErrchk( (cudaMalloc(&(array.x[j]), floatBytes) ));
	}
	array.setCapacity(width);
}

template<our_size_t DIM, typename Float> void copyDeviceVecArray(our_size_t width, const VecArray<DIM, Float>& dst, const VecArray<DIM, Float>& src, cudaMemcpyKind dir){
	// We know how big the tree is now. Don't make extra space
	our_size_t floatBytes = width*sizeof(Float);
	//printf("copying array " SZSTR " with dir %d:\n",width,dir);
	for(our_size_t j = 0; j < DIM; j++){
		//printf("\t%p\t%p\n",dst.x[j],src.x[j]);
		gpuErrchk( (cudaMemcpy(dst.x[j], src.x[j], floatBytes, dir)) );
	}
}


template<our_size_t DIM, typename Float> void freeDeviceVecArray(VecArray<DIM, Float>& array){
	for(our_size_t j = 0; j < DIM; j++){
		gpuErrchk( (cudaFree(array.x[j])) );
	}
}


//---------------------------------------------------
//---------------------------------------------------



template<our_size_t DIM, typename Float> void allocDevicePointMassArray(our_size_t width, PointMassArray<DIM, Float>& masses){

	our_size_t floatBytes = width*sizeof(Float);
	ALLOC_DEBUG_MSG(floatBytes);
	allocDeviceVecArray(width, masses.pos);
	gpuErrchk( (cudaMalloc(&(masses.m), floatBytes) ));
	masses.setCapacity(width);

}

template<our_size_t DIM, typename Float> void copyDevicePointMassArray(our_size_t width, PointMassArray<DIM, Float>& dst, const PointMassArray<DIM, Float>& src, cudaMemcpyKind dir){

	// We know how big the tree is now. Don't make extra space
	our_size_t floatBytes = width*sizeof(Float);
	copyDeviceVecArray(width, dst.pos, src.pos, dir);
	gpuErrchk( (cudaMemcpy(dst.m, src.m, floatBytes, dir)) );

}


template<our_size_t DIM, typename Float> void freeDevicePointMassArray(PointMassArray<DIM, Float>& array){
	freeDeviceVecArray(array.pos);
	gpuErrchk( (cudaFree(array.m)) );
}


//---------------------------------------------------
//---------------------------------------------------

template<our_size_t DIM, typename Float> void allocDeviceNodeArray(our_size_t width, NodeArray<DIM, Float>& level){
	
	// We know how big the tree is now. Don't make extra space
	
	ALLOC_DEBUG_MSG(width * (sizeof(bool) + 2 * sizeof(our_size_t) + sizeof(Float)));
	

	
	gpuErrchk( (cudaMalloc(&(level.isLeaf), width*sizeof(bool)) ));
	
	our_size_t countBytes = width*sizeof(our_size_t);
	gpuErrchk( (cudaMalloc(&(level.childCount),  countBytes)) );
	gpuErrchk( (cudaMalloc(&(level.childStart), countBytes)) );
	
	our_size_t floatBytes = width*sizeof(Float);
	allocDeviceVecArray(width, level.minX);
	allocDeviceVecArray(width, level.maxX);
	allocDevicePointMassArray(width, level.barycenter);
	
	gpuErrchk( (cudaMalloc(&(level.radius), floatBytes) ));
	level.setCapacity(width);
	
}

template<our_size_t DIM, typename Float> void copyDeviceNodeArray(our_size_t width, NodeArray<DIM, Float>& level, const NodeArray<DIM, Float>& src, cudaMemcpyKind dir){
	
	// We know how big the tree is now. Don't make extra space
	gpuErrchk( (cudaMemcpy(level.isLeaf, src.isLeaf, width*sizeof(bool), dir)) );
	
	our_size_t countBytes = width*sizeof(our_size_t);
	gpuErrchk( (cudaMemcpy(level.childCount, src.childCount, countBytes, dir)) );
	gpuErrchk( (cudaMemcpy(level.childStart, src.childStart, countBytes, dir)) );
	/*
	for(our_size_t i = 0; i < width; i++){
		printf("(" SZSTR " " SZSTR ")\t",src.childCount[i],src.childStart[i]);
	 }
	 printf("\n");
	 //*/
	
	
	
	our_size_t floatBytes = width*sizeof(Float);
	
	copyDeviceVecArray(width, level.minX, src.minX, dir);
	copyDeviceVecArray(width, level.maxX, src.maxX, dir);
	copyDevicePointMassArray(width, level.barycenter, src.barycenter, dir);
	
	gpuErrchk( (cudaMemcpy(level.radius, src.radius, floatBytes, dir)) );
	
}


template<our_size_t DIM, typename Float> void freeDeviceNodeArray(NodeArray<DIM, Float>& array){
	gpuErrchk( (cudaFree(array.isLeaf)) );
	gpuErrchk( (cudaFree(array.childCount)) );
	gpuErrchk( (cudaFree(array.childStart)) );
	
	freeDeviceVecArray(array.minX);
	freeDeviceVecArray(array.maxX);
	freeDevicePointMassArray(array.barycenter);
	
	gpuErrchk( (cudaFree(array.radius)) );
}


template<our_size_t DIM, typename Float, our_size_t MAX_LEVELS>
void makeDeviceTree(NodeArray<DIM, Float> treeLevels[MAX_LEVELS], NodeArray<DIM, Float> placeHolderTree[MAX_LEVELS], our_size_t treeCounts[MAX_LEVELS]){
	
	for(our_size_t i = 0; i < MAX_LEVELS; i++){
		NodeArray<DIM, Float> level;
		
		//printf("Copying level:" SZSTR "\n\t",i);
		
		allocDeviceNodeArray(treeCounts[i], level);
		copyDeviceNodeArray(treeCounts[i], level, treeLevels[i], cudaMemcpyHostToDevice);
		/*
		for(our_size_t j = 0; j < treeCounts[i]; j++){
			printf("(" SZSTR " " SZSTR ")\t",treeLevels[i].childCount[j], treeLevels[i].childStart[j]);
		} printf("\n");
		 //*/
		placeHolderTree[i] = level;
	}
}

template<our_size_t DIM, typename Float, our_size_t MAX_LEVELS>
void freeDeviceTree(NodeArray<DIM, Float> placeHolderTree[MAX_LEVELS]){
	
	for(our_size_t i = 0; i < MAX_LEVELS; i++){
		freeDeviceNodeArray(placeHolderTree[i]);
	}
}

//---------------------------------------------------
//---------------------------------------------------



template<our_size_t DIM, typename Float> void allocDeviceParticleArray(our_size_t width, ParticleArray<DIM, Float>& particles){
	
	allocDevicePointMassArray(width, particles.mass);
	allocDeviceVecArray(width, particles.vel);
	particles.setCapacity(width);
	
}

template<our_size_t DIM, typename Float> void copyDeviceParticleArray(our_size_t width, ParticleArray<DIM, Float>& dst, const ParticleArray<DIM, Float>& src, cudaMemcpyKind dir){
	
	// We know how big the tree is now. Don't make extra space
	copyDevicePointMassArray(width, dst.mass, src.mass, dir);
	copyDeviceVecArray(width, dst.vel, src.vel, dir);
	
}


template<our_size_t DIM, typename Float> void freeDeviceParticleArray(ParticleArray<DIM, Float>& array){
	freeDevicePointMassArray(array.mass);
	freeDeviceVecArray(array.vel);
}

//---------------------------------------------------
//---------------------------------------------------



template<our_size_t DIM, typename Float, our_size_t MAX_PARTS> void allocDeviceGroupInfoArray(our_size_t width, GroupInfoArray<DIM, Float, MAX_PARTS>& group){
	
	
	ALLOC_DEBUG_MSG(width * (2 * sizeof(our_size_t) + sizeof(Float)));
	
	our_size_t countBytes = width*sizeof(our_size_t);
	gpuErrchk( (cudaMalloc(&(group.childCount),  countBytes)) );
	gpuErrchk( (cudaMalloc(&(group.childStart), countBytes)) );
	
	our_size_t floatBytes = width*sizeof(Float);
	allocDeviceVecArray(width, group.minX);
	allocDeviceVecArray(width, group.maxX);
	allocDeviceVecArray(width, group.center);
	
	gpuErrchk( (cudaMalloc(&(group.radius), floatBytes) ));
	group.setCapacity(width);
	
}

template<our_size_t DIM, typename Float, our_size_t MAX_PARTS> void copyDeviceGroupInfoArray(our_size_t width, GroupInfoArray<DIM, Float, MAX_PARTS>& dst, const GroupInfoArray<DIM, Float, MAX_PARTS>& src, cudaMemcpyKind dir){
	
	// We know how big the tree is now. Don't make extra space
	our_size_t countBytes = width*sizeof(our_size_t);
	gpuErrchk( (cudaMemcpy(dst.childCount, src.childCount, countBytes, dir)) );
	gpuErrchk( (cudaMemcpy(dst.childStart, src.childStart, countBytes, dir)) );
	/*
	for(our_size_t i = 0; i < width; i++){
		printf("" SZSTR "\t",src.childCount[i]);
	}
	printf("\n");
	 //*/
	
	our_size_t floatBytes = width*sizeof(Float);
	copyDeviceVecArray(width, dst.minX, src.minX, dir);
	copyDeviceVecArray(width, dst.maxX, src.maxX, dir);
	copyDeviceVecArray(width, dst.center, src.center, dir);
	gpuErrchk( (cudaMemcpy(dst.radius, src.radius, floatBytes, dir)) );
	
}


template<our_size_t DIM, typename Float, our_size_t MAX_PARTS> void freeDeviceGroupInfoArray(GroupInfoArray<DIM, Float, MAX_PARTS>& array){
	gpuErrchk( (cudaFree(array.childCount)) );
	gpuErrchk( (cudaFree(array.childStart)) );
	freeDeviceVecArray(array.minX);
	freeDeviceVecArray(array.maxX);
	freeDeviceVecArray(array.center);
	gpuErrchk( (cudaFree(array.radius)) );
}

//---------------------------------------------------

#endif
