//
//  TransposalTools.h
//  TreeTraverse
//
//  Created by Thomas Dickerson on 3/16/15.
//  Copyright (c) 2015 StickFigure Graphic Productions. All rights reserved.
//

#ifndef TreeTraverse_TransposalTools_h
#define TreeTraverse_TransposalTools_h
#include "treedefs.h"
#include <iostream>
#include <vector>

//---------------------------------------------------



template<our_size_t DIM, typename Float> void allocVecArray(our_size_t width, VecArray<DIM, Float>& array){
	for(our_size_t j = 0; j < DIM; j++){
		array.x[j] = new Float[width];
	}
	array.setCapacity(width);
	
}

template<our_size_t DIM, typename Float> void copyVecArray(our_size_t width, const VecArray<DIM, Float>& dst, const VecArray<DIM, Float>& src){
	// We know how big the tree is now. Don't make extra space
	our_size_t floatBytes = width*sizeof(Float);
	for(our_size_t j = 0; j < DIM; j++){
		memcpy(dst.x[j], src.x[j], floatBytes);
	}
}


template<our_size_t DIM, typename Float> void freeVecArray(VecArray<DIM, Float>& array){
	for(our_size_t j = 0; j < DIM; j++){
		delete [] array.x[j];
	}
}

//---------------------------------------------------
//---------------------------------------------------

template<our_size_t DIM, typename Float> void allocNodeArray(our_size_t width, NodeArray<DIM, Float>& level){
	
	// We know how big the tree is now. Don't make extra space
	level.isLeaf = new bool[width];
	level.childCount = new our_size_t[width];
	level.childStart = new our_size_t[width];
	
	allocVecArray(width, level.minX);
	allocVecArray(width, level.maxX);
	allocPointMassArray(width, level.barycenter);
	
	level.radius = new Float[width];
	level.setCapacity(width);
}

template<our_size_t DIM, typename Float> void copyNodeArray(our_size_t width, NodeArray<DIM, Float>& level, const NodeArray<DIM, Float>& src){
	
	// We know how big the tree is now. Don't make extra space
	memcpy(level.isLeaf, src.isLeaf, width*sizeof(bool));
	our_size_t countBytes = width*sizeof(our_size_t);
	memcpy(level.childCount, src.childCount, countBytes);
	memcpy(level.childStart, src.childStart, countBytes);
	
	
	
	our_size_t floatBytes = width*sizeof(Float);
	
	copyVecArray(width, level.minX, src.minX);
	copyVecArray(width, level.maxX, src.maxX);
	copyPointMassArray(width, level.barycenter, src.barycenter);
	
	memcpy(level.radius, src.radius, floatBytes);
	
}


template<our_size_t DIM, typename Float> void freeNodeArray(NodeArray<DIM, Float>& array){
	delete [] array.isLeaf;
	delete [] array.childCount;
	delete [] array.childStart;
	
	
	freeVecArray(array.minX);
	freeVecArray(array.maxX);
	freePointMassArray(array.barycenter);
	
	delete [] array.radius;
}


template<our_size_t DIM, typename Float, our_size_t MAX_LEVELS>
void duplicateArrayTree(NodeArray<DIM, Float> treeLevels[MAX_LEVELS], NodeArray<DIM, Float> placeHolderTree[MAX_LEVELS], our_size_t treeCounts[MAX_LEVELS]){
	
	for(our_size_t i = 0; i < MAX_LEVELS; i++){
		NodeArray<DIM, Float> level;
		
		allocNodeArray(treeCounts[i], level);
		copyNodeArray(treeCounts[i], level, treeLevels[i]);
		
		placeHolderTree[i] = level;
	}
}

template<our_size_t DIM, typename Float, our_size_t MAX_LEVELS>
void freeArrayTree(NodeArray<DIM, Float> placeHolderTree[MAX_LEVELS]){
	
	for(our_size_t i = 0; i < MAX_LEVELS; i++){
		freeNodeArray(placeHolderTree[i]);
	}
}


//---------------------------------------------------
//---------------------------------------------------



template<our_size_t DIM, typename Float> void allocPointMassArray(our_size_t width, PointMassArray<DIM, Float>& masses){
	
	allocVecArray(width, masses.pos);
	masses.m = new Float[width];
	masses.setCapacity(width);
	
}

template<our_size_t DIM, typename Float> void copyPointMassArray(our_size_t width, PointMassArray<DIM, Float>& dst, const PointMassArray<DIM, Float>& src){
	
	// We know how big the tree is now. Don't make extra space
	our_size_t floatBytes = width*sizeof(Float);
	copyVecArray(width, src.pos, dst.pos);
	memcpy(dst.m, src.m, floatBytes);
	
}


template<our_size_t DIM, typename Float> void freePointMassArray(PointMassArray<DIM, Float>& array){
	freeVecArray(array.pos);
	delete [] array.m;
}

//---------------------------------------------------
//---------------------------------------------------



template<our_size_t DIM, typename Float> void allocParticleArray(our_size_t width, ParticleArray<DIM, Float>& particles){
	
	allocPointMassArray(width, particles.mass);
	allocVecArray(width, particles.vel);
	particles.setCapacity(width);
	
}

template<our_size_t DIM, typename Float> void copyParticleArray(our_size_t width, ParticleArray<DIM, Float>& dst, const ParticleArray<DIM, Float>& src){
	
	// We know how big the tree is now. Don't make extra space
	copyPointMassArray(width, src.mass, dst.mass);
	copyVecArray(width, src.vel, dst.vel);
	
}


template<our_size_t DIM, typename Float> void freeParticleArray(ParticleArray<DIM, Float>& array){
	freePointMassArray(array.mass);
	freeVecArray(array.vel);
}

//---------------------------------------------------
//---------------------------------------------------



template<our_size_t DIM, typename Float, our_size_t MAX_PARTS> void allocGroupInfoArray(our_size_t width, GroupInfoArray<DIM, Float, MAX_PARTS>& group){
	
	group.childCount = new our_size_t[width];
	group.childStart = new our_size_t[width];
	
	allocVecArray(width, group.minX);
	allocVecArray(width, group.maxX);
	allocVecArray(width, group.center);
	
	group.radius = new Float[width];
	group.setCapacity(width);
}

template<our_size_t DIM, typename Float, our_size_t MAX_PARTS> void copyGroupInfoArray(our_size_t width, GroupInfoArray<DIM, Float, MAX_PARTS>& dst, const GroupInfoArray<DIM, Float, MAX_PARTS>& src){
	
	// We know how big the tree is now. Don't make extra space
	our_size_t countBytes = width*sizeof(our_size_t);
	memcpy(dst.childCount, src.childCount, countBytes);
	memcpy(dst.childStart, src.childStart, countBytes);
	
	
	our_size_t floatBytes = width*sizeof(Float);
	copyVecArray(width, src.minX, dst.minX);
	copyVecArray(width, src.maxX, dst.maxX);
	copyVecArray(width, src.center, dst.center);
	
	memcpy(dst.radius, src.radius, floatBytes);
	
}


template<our_size_t DIM, typename Float, our_size_t MAX_PARTS> void freeGroupInfoArray(GroupInfoArray<DIM, Float, MAX_PARTS>& array){
	
	delete [] array.childCount;
	delete [] array.childStart;
	freeVecArray(array.minX);
	freeVecArray(array.maxX);
	freeVecArray(array.center);
	delete [] array.radius;
}

//---------------------------------------------------


// VecArray must be preallocated
template<our_size_t DIM, typename Float>
VecArray<DIM, Float> pArrayFromPVec(const std::vector< Particle<DIM, Float> >& vec, ParticleArray<DIM, Float> array){
	
	for(our_size_t i = 0; i < vec.size(); i++){
		array[i] = vec[i];
	}
	
}

#endif
