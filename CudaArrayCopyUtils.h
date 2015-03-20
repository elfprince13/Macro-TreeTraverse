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

#define ALLOC_DEBUG_MSG(N) 	printf("%s @ %s:%d: Allocating %lu bytes\n",__func__,__FILE__,__LINE__,N)

//---------------------------------------------------



template<size_t DIM, typename Float> void allocDeviceVecArray(size_t width, VecArray<DIM, Float>& array){
	size_t floatBytes = width*sizeof(Float);
	ALLOC_DEBUG_MSG(floatBytes*DIM);
	for(size_t j = 0; j < DIM; j++){
		gpuErrchk( (cudaMalloc(&(array.x[j]), floatBytes) ));
	}
	array.setCapacity(width);
}

template<size_t DIM, typename Float> void copyDeviceVecArray(size_t width, const VecArray<DIM, Float>& dst, const VecArray<DIM, Float>& src, cudaMemcpyKind dir){
	// We know how big the tree is now. Don't make extra space
	size_t floatBytes = width*sizeof(Float);
	//printf("copying array %lu with dir %d:\n",width,dir);
	for(size_t j = 0; j < DIM; j++){
		//printf("\t%p\t%p\n",dst.x[j],src.x[j]);
		gpuErrchk( (cudaMemcpy(dst.x[j], src.x[j], floatBytes, dir)) );
	}
}


template<size_t DIM, typename Float> void freeDeviceVecArray(VecArray<DIM, Float>& array){
	for(size_t j = 0; j < DIM; j++){
		gpuErrchk( (cudaFree(array.x[j])) );
	}
}


//---------------------------------------------------
//---------------------------------------------------



template<size_t DIM, typename Float> void allocDevicePointMassArray(size_t width, PointMassArray<DIM, Float>& masses){

	size_t floatBytes = width*sizeof(Float);
	ALLOC_DEBUG_MSG(floatBytes);
	allocDeviceVecArray(width, masses.pos);
	gpuErrchk( (cudaMalloc(&(masses.m), floatBytes) ));
	masses.setCapacity(width);

}

template<size_t DIM, typename Float> void copyDevicePointMassArray(size_t width, PointMassArray<DIM, Float>& dst, const PointMassArray<DIM, Float>& src, cudaMemcpyKind dir){

	// We know how big the tree is now. Don't make extra space
	size_t floatBytes = width*sizeof(Float);
	copyDeviceVecArray(width, dst.pos, src.pos, dir);
	gpuErrchk( (cudaMemcpy(dst.m, src.m, floatBytes, dir)) );

}


template<size_t DIM, typename Float> void freeDevicePointMassArray(PointMassArray<DIM, Float>& array){
	freeDeviceVecArray(array.pos);
	gpuErrchk( (cudaFree(array.m)) );
}


//---------------------------------------------------
//---------------------------------------------------

template<size_t DIM, typename Float> void allocDeviceNodeArray(size_t width, NodeArray<DIM, Float>& level){
	
	// We know how big the tree is now. Don't make extra space
	
	ALLOC_DEBUG_MSG(width * (sizeof(bool) + 2 * sizeof(size_t) + sizeof(Float)));
	

	
	gpuErrchk( (cudaMalloc(&(level.isLeaf), width*sizeof(bool)) ));
	
	size_t countBytes = width*sizeof(size_t);
	gpuErrchk( (cudaMalloc(&(level.childCount),  countBytes)) );
	gpuErrchk( (cudaMalloc(&(level.childStart), countBytes)) );
	
	size_t floatBytes = width*sizeof(Float);
	allocDeviceVecArray(width, level.minX);
	allocDeviceVecArray(width, level.maxX);
	allocDevicePointMassArray(width, level.barycenter);
	
	gpuErrchk( (cudaMalloc(&(level.radius), floatBytes) ));
	level.setCapacity(width);
	
}

template<size_t DIM, typename Float> void copyDeviceNodeArray(size_t width, NodeArray<DIM, Float>& level, const NodeArray<DIM, Float>& src, cudaMemcpyKind dir){
	
	// We know how big the tree is now. Don't make extra space
	gpuErrchk( (cudaMemcpy(level.isLeaf, src.isLeaf, width*sizeof(bool), dir)) );
	
	size_t countBytes = width*sizeof(size_t);
	gpuErrchk( (cudaMemcpy(level.childCount, src.childCount, countBytes, dir)) );
	gpuErrchk( (cudaMemcpy(level.childStart, src.childStart, countBytes, dir)) );
	/*
	for(size_t i = 0; i < width; i++){
		printf("(%lu %lu)\t",src.childCount[i],src.childStart[i]);
	 }
	 printf("\n");
	 //*/
	
	
	
	size_t floatBytes = width*sizeof(Float);
	
	copyDeviceVecArray(width, level.minX, src.minX, dir);
	copyDeviceVecArray(width, level.maxX, src.maxX, dir);
	copyDevicePointMassArray(width, level.barycenter, src.barycenter, dir);
	
	gpuErrchk( (cudaMemcpy(level.radius, src.radius, floatBytes, dir)) );
	
}


template<size_t DIM, typename Float> void freeDeviceNodeArray(NodeArray<DIM, Float>& array){
	gpuErrchk( (cudaFree(array.isLeaf)) );
	gpuErrchk( (cudaFree(array.childCount)) );
	gpuErrchk( (cudaFree(array.childStart)) );
	
	freeDeviceVecArray(array.minX);
	freeDeviceVecArray(array.maxX);
	freeDevicePointMassArray(array.barycenter);
	
	gpuErrchk( (cudaFree(array.radius)) );
}


template<size_t DIM, typename Float, size_t MAX_LEVELS>
void makeDeviceTree(NodeArray<DIM, Float> treeLevels[MAX_LEVELS], NodeArray<DIM, Float> placeHolderTree[MAX_LEVELS], size_t treeCounts[MAX_LEVELS]){
	
	for(size_t i = 0; i < MAX_LEVELS; i++){
		NodeArray<DIM, Float> level;
		
		//printf("Copying level:%lu\n\t",i);
		
		allocDeviceNodeArray(treeCounts[i], level);
		copyDeviceNodeArray(treeCounts[i], level, treeLevels[i], cudaMemcpyHostToDevice);
		/*
		for(size_t j = 0; j < treeCounts[i]; j++){
			printf("(%lu %lu)\t",treeLevels[i].childCount[j], treeLevels[i].childStart[j]);
		} printf("\n");
		 //*/
		placeHolderTree[i] = level;
	}
}

template<size_t DIM, typename Float, size_t MAX_LEVELS>
void freeDeviceTree(NodeArray<DIM, Float> placeHolderTree[MAX_LEVELS]){
	
	for(size_t i = 0; i < MAX_LEVELS; i++){
		freeDeviceNodeArray(placeHolderTree[i]);
	}
}

//---------------------------------------------------
//---------------------------------------------------



template<size_t DIM, typename Float> void allocDeviceParticleArray(size_t width, ParticleArray<DIM, Float>& particles){
	
	allocDevicePointMassArray(width, particles.mass);
	allocDeviceVecArray(width, particles.vel);
	particles.setCapacity(width);
	
}

template<size_t DIM, typename Float> void copyDeviceParticleArray(size_t width, ParticleArray<DIM, Float>& dst, const ParticleArray<DIM, Float>& src, cudaMemcpyKind dir){
	
	// We know how big the tree is now. Don't make extra space
	copyDevicePointMassArray(width, dst.mass, src.mass, dir);
	copyDeviceVecArray(width, dst.vel, src.vel, dir);
	
}


template<size_t DIM, typename Float> void freeDeviceParticleArray(ParticleArray<DIM, Float>& array){
	freeDevicePointMassArray(array.mass);
	freeDeviceVecArray(array.vel);
}

//---------------------------------------------------
//---------------------------------------------------



template<size_t DIM, typename Float, size_t MAX_PARTS> void allocDeviceGroupInfoArray(size_t width, GroupInfoArray<DIM, Float, MAX_PARTS>& group){
	
	
	ALLOC_DEBUG_MSG(width * (2 * sizeof(size_t) + sizeof(Float)));
	
	size_t countBytes = width*sizeof(size_t);
	gpuErrchk( (cudaMalloc(&(group.childCount),  countBytes)) );
	gpuErrchk( (cudaMalloc(&(group.childStart), countBytes)) );
	
	size_t floatBytes = width*sizeof(Float);
	allocDeviceVecArray(width, group.minX);
	allocDeviceVecArray(width, group.maxX);
	allocDeviceVecArray(width, group.center);
	
	gpuErrchk( (cudaMalloc(&(group.radius), floatBytes) ));
	group.setCapacity(width);
	
}

template<size_t DIM, typename Float, size_t MAX_PARTS> void copyDeviceGroupInfoArray(size_t width, GroupInfoArray<DIM, Float, MAX_PARTS>& dst, const GroupInfoArray<DIM, Float, MAX_PARTS>& src, cudaMemcpyKind dir){
	
	// We know how big the tree is now. Don't make extra space
	size_t countBytes = width*sizeof(size_t);
	gpuErrchk( (cudaMemcpy(dst.childCount, src.childCount, countBytes, dir)) );
	gpuErrchk( (cudaMemcpy(dst.childStart, src.childStart, countBytes, dir)) );
	/*
	for(size_t i = 0; i < width; i++){
		printf("%lu\t",src.childCount[i]);
	}
	printf("\n");
	 //*/
	
	size_t floatBytes = width*sizeof(Float);
	copyDeviceVecArray(width, dst.minX, src.minX, dir);
	copyDeviceVecArray(width, dst.maxX, src.maxX, dir);
	copyDeviceVecArray(width, dst.center, src.center, dir);
	gpuErrchk( (cudaMemcpy(dst.radius, src.radius, floatBytes, dir)) );
	
}


template<size_t DIM, typename Float, size_t MAX_PARTS> void freeDeviceGroupInfoArray(GroupInfoArray<DIM, Float, MAX_PARTS>& array){
	gpuErrchk( (cudaFree(array.childCount)) );
	gpuErrchk( (cudaFree(array.childStart)) );
	freeDeviceVecArray(array.minX);
	freeDeviceVecArray(array.maxX);
	freeDeviceVecArray(array.center);
	gpuErrchk( (cudaFree(array.radius)) );
}

//---------------------------------------------------

#endif
