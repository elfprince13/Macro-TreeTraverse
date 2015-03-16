// We want to transform AOS into SOA wherever possible!
// This is something we can do with a macro =)

#ifndef _TreeTraverse_TreeDefs_h_
#define _TreeTraverse_TreeDefs_h_

#ifdef __CUDA_ARCH__
#include <cuda.h>
#include <device_functions.h>
#define UNIVERSAL_STORAGE __host__ __device__
#else
#include <cstddef>
#include <cmath>
#define UNIVERSAL_STORAGE
#endif

typedef unsigned short uint16;

template<size_t DIM, typename T> struct _ArrayVecProxy;

template<size_t DIM, typename T> struct Vec{
	T x[DIM];

	UNIVERSAL_STORAGE Vec<DIM, T>(){}
	UNIVERSAL_STORAGE Vec<DIM, T>(const _ArrayVecProxy<DIM, T> n){
		for(size_t i = 0; i < DIM; i++){
			x[i] = *(n.x[i]);
		}
	}
	
	UNIVERSAL_STORAGE inline Vec<DIM, T> operator -(const Vec<DIM, T> &v) const{
		Vec<DIM, T> out;
		for(size_t i = 0; i < DIM; i++){
			out.x[i] = this->x[i] - v.x[i];
		}
		return out;
	}
	
	UNIVERSAL_STORAGE inline Vec<DIM, bool> operator <(const Vec<DIM, T> &v) const{
		Vec<DIM, bool> out;
		for(size_t i = 0; i < DIM; i++){
			out.x[i] = this->x[i] < v.x[i];
		}
		return out;
	}
	
	UNIVERSAL_STORAGE inline Vec<DIM, T> operator +(const Vec<DIM, T> &v) const{
		Vec<DIM, T> out;
		for(size_t i = 0; i < DIM; i++){
			out.x[i] = this->x[i] + v.x[i];
		}
		return out;
	}
	
	UNIVERSAL_STORAGE inline Vec<DIM, T> operator *(T s) const{
		Vec<DIM, T> out;
		for(size_t i = 0; i < DIM; i++){
			out.x[i] = this->x[i] * s;
		}
		return out;
	}
	
	UNIVERSAL_STORAGE inline Vec<DIM, T> operator /(T s) const{
		Vec<DIM, T> out;
		for(size_t i = 0; i < DIM; i++){
			out.x[i] = this->x[i] / s;
		}
		return out;
	}
	
	UNIVERSAL_STORAGE inline Vec<DIM, T> operator *(const Vec<DIM, T> &v) const{
		Vec<DIM, T> out;
		for(size_t i = 0; i < DIM; i++){
			out.x[i] = this->x[i] * v.x[i];
		}
		return out;
	}
	
	UNIVERSAL_STORAGE inline Vec<DIM, T> operator /(const Vec<DIM, T> &v) const{
		Vec<DIM, T> out;
		for(size_t i = 0; i < DIM; i++){
			out.x[i] = this->x[i] / v.x[i];
		}
		return out;
	}
};

template<size_t DIM, typename T> struct _ArrayVecProxy{
	T* x[DIM];
	UNIVERSAL_STORAGE inline _ArrayVecProxy<DIM, T>& operator=(const Vec<DIM, T> &v) {
		for(size_t i = 0; i < DIM; i++){
			*(x[i]) = v.x[i];
		}
		return *this;
	}
	
};

template <size_t DIM, typename T> struct VecArray{
	T *x[DIM];
	
	UNIVERSAL_STORAGE inline Vec<DIM, T> operator[](size_t i) const{
		Vec<DIM, T> t;
		for(size_t j = 0; j < DIM; j++){
			t.x[j] = x[j][i];
		}
		return t;
	}
	
	
	UNIVERSAL_STORAGE inline _ArrayVecProxy<DIM, T> operator[](size_t i) {
		_ArrayVecProxy<DIM, T> t;
		for(size_t j = 0; j < DIM; j++){
			t.x[j] = &(x[j][i]);
		}
		return t;
	}
	
	UNIVERSAL_STORAGE inline VecArray<DIM, T> operator +(size_t i) const {
		VecArray<DIM, T> o;
		for(size_t j = 0; j < DIM; j++){
			o.x[j] = x[j] + i;
		}
		return o;
	}
};

/*
// we require DIM*sizeof(TT) < sizeof(OT) or this will be undefined.
// Doing this properly with templates is a pain.
// Also require min(mN,mX) == mN && max(mN,mX) == mX
template<size_t DIM, typename T, typename TT, typename OT>
OT zIndex(Vec<DIM, T> v, Vec<DIM, T> mN, Vec<DIM, T> mX) {
	OT z = 0;
	Vec<DIM, T> ones; for(size_t i = 0; i < DIM; i++){ ones.x[i] = 1; }
	Vec<DIM, T> mXSafe = max(ones, mX - mN);
	Vec<DIM, T> nv = ((v - mN) / mXSafe) * (1 << sizeof(TT));
	Vec<DIM, TT> bv; for(size_t i = 0; i < DIM; i++){ bv.x[i] = (TT) nv.x[i]; }
	for(size_t b = 0; b < sizeof(TT); b++){
		for(size_t i = 0; i < DIM; i++){
			OT flag = ((bv.x[i] & (1 << b)) >> b);
			z |= flag << (DIM * b + i);
		}
	}
	return z;
	
}
 */


template<size_t DIM, typename T> UNIVERSAL_STORAGE bool contains(const Vec<DIM, T> &lower, const Vec<DIM, T> &upper, const Vec<DIM, T> point){
	bool is_contained = true;
	for(size_t i = 0; is_contained && i < DIM; i++){
		is_contained &= (lower.x[i] <= point.x[i]) && (upper.x[i] >= point.x[i]);
	}
	return is_contained;
}

template<size_t DIM, typename T> UNIVERSAL_STORAGE Vec<DIM, T> min(const Vec<DIM, T> &v1, const Vec<DIM, T> &v2){
	Vec<DIM, T> ret;
	for(size_t i = 0; i < DIM; i++){
		ret.x[i] = (v1.x[i] < v2.x[i]) ? v1.x[i] : v2.x[i];
	}
	return ret;
}


template<size_t DIM, typename T> UNIVERSAL_STORAGE Vec<DIM, T> max(const Vec<DIM, T> &v1, const Vec<DIM, T> &v2){
	Vec<DIM, T> ret;
	for(size_t i = 0; i < DIM; i++){
		ret.x[i] = (v1.x[i] > v2.x[i]) ? v1.x[i] : v2.x[i];
	}
	return ret;
}


template<size_t DIM, typename T> UNIVERSAL_STORAGE T mag_sq(const Vec<DIM, T> &v){
	T ret = 0;
	for(size_t i = 0; i < DIM; i++){
		ret += v.x[i] * v.x[i];
	}
	return ret;
}


template<size_t DIM, typename T> UNIVERSAL_STORAGE T mag(const Vec<DIM, T> &v){
	return (T) sqrt(mag_sq(v));
}

template<size_t DIM, typename T, size_t MAX_PARTS> struct GroupInfo{
	size_t childCount;
	size_t childStart;
	Vec<DIM, T> minX;
	Vec<DIM, T> maxX;
	Vec<DIM, T> center;
	T radius;
};

template<size_t DIM, typename T, size_t MAX_PARTS> struct _ArrayGroupInfoProxy{
	size_t *childCount;
	size_t *childStart;
	_ArrayVecProxy<DIM, T> minX;
	_ArrayVecProxy<DIM, T> maxX;
	_ArrayVecProxy<DIM, T> center;
	T *radius;
	
	UNIVERSAL_STORAGE inline _ArrayGroupInfoProxy<DIM, T, MAX_PARTS>& operator=(const GroupInfo<DIM, T, MAX_PARTS> &v) {
		*childCount = v.childCount;
		*childStart = v.childStart;
		minX = v.minX;
		maxX = v.maxX;
		center = v.center;
		*radius = v.radius;
		
		return *this;
	}
};

template<size_t DIM, typename T, size_t MAX_PARTS> struct GroupInfoArray{
	size_t *childCount;
	size_t *childStart;
	VecArray<DIM, T> minX;
	VecArray<DIM, T> maxX;
	VecArray<DIM, T> center;
	T *radius;
	
	UNIVERSAL_STORAGE inline GroupInfo<DIM, T, MAX_PARTS> operator [](size_t i) const {
		GroupInfo<DIM, T, MAX_PARTS> t;
		t.childCount = childCount[i];
		t.childStart = childStart[i];
		t.minX = minX[i];
		t.maxX = maxX[i];
		t.center = center[i];
		t.radius = radius[i];
		
		return t;
	}
	
	
	UNIVERSAL_STORAGE inline _ArrayGroupInfoProxy<DIM, T, MAX_PARTS> operator [](size_t i) {
		_ArrayGroupInfoProxy<DIM, T, MAX_PARTS> t;
		t.childCount = &(childCount[i]);
		t.childStart = &(childStart[i]);
		t.minX = minX[i];
		t.maxX = maxX[i];
		t.center = center[i];
		t.radius = &(radius[i]);
		return t;
	}
	
	UNIVERSAL_STORAGE inline GroupInfoArray<DIM, T, MAX_PARTS> operator +(size_t i) const {
		GroupInfoArray<DIM, T, MAX_PARTS> o;
		o.childCount = childCount + i;
		o.childStart = childStart + i;
		o.minX = minX + i;
		o.maxX = maxX + i;
		o.center = center + i;
		o.radius = radius + i;
		return o;
	}
	
	
};

template<size_t DIM, typename T> struct _ArrayNodeProxy;

template<size_t DIM, typename T> struct Node{
	bool isLeaf;
	size_t childCount;
	size_t childStart;
	Vec<DIM, T> minX;
	Vec<DIM, T> maxX;
	Vec<DIM, T> barycenter;
	T mass;
	T radius;
	
	UNIVERSAL_STORAGE Node<DIM, T>(){}
	UNIVERSAL_STORAGE Node<DIM, T>(const _ArrayNodeProxy<DIM, T> n){
		isLeaf = *n.isLeaf;
		childCount = *n.childCount;
		childStart = *n.childStart;
		minX = n.minX;
		maxX = n.maxX;
		barycenter = n.barycenter;
		mass = *n.mass;
		radius = *n.radius;
	}
};

template<size_t DIM, typename T> struct _ArrayNodeProxy{
	bool *isLeaf;
	size_t *childCount;
	size_t *childStart;
	_ArrayVecProxy<DIM, T> minX;
	_ArrayVecProxy<DIM, T> maxX;
	_ArrayVecProxy<DIM, T> barycenter;
	T *mass;
	T *radius;
	
	UNIVERSAL_STORAGE inline _ArrayNodeProxy<DIM, T>& operator=(const Node<DIM, T> &v) {
		*isLeaf = v.isLeaf;
		*childCount = v.childCount;
		*childStart = v.childStart;
		minX = v.minX;
		maxX = v.maxX;
		barycenter = v.barycenter;
		*mass = v.mass;
		*radius = v.radius;
		return *this;
	}
};

template<size_t DIM, typename T> struct NodeArray{
	bool *isLeaf;
	size_t *childCount;
	size_t *childStart;
	VecArray<DIM, T> minX;
	VecArray<DIM, T> maxX;
	VecArray<DIM, T> barycenter;
	T *mass;
	T *radius;
	
	UNIVERSAL_STORAGE NodeArray<DIM, T>(){}
	UNIVERSAL_STORAGE NodeArray<DIM, T>(Node<DIM, T>& n){
		isLeaf = &(n.isLeaf);
		childCount = &(n.childCount);
		childStart = &(n.childStart);
		
		mass = &(n.mass);
		radius = &(n.radius);
	}
	
	UNIVERSAL_STORAGE inline Node<DIM, T> operator [](size_t i) const {
		Node<DIM, T> t;
		t.isLeaf = isLeaf[i];
		t.childCount = childCount[i];
		t.childStart = childStart[i];
		t.minX = minX[i];
		t.maxX = maxX[i];
		t.barycenter = barycenter[i];
		t.mass = mass[i];
		t.radius = radius[i];
		
		return t;
	}
	
	
	UNIVERSAL_STORAGE inline _ArrayNodeProxy<DIM, T> operator [](size_t i) {
		_ArrayNodeProxy<DIM, T> t;
		t.isLeaf = &(isLeaf[i]);
		t.childCount = &(childCount[i]);
		t.childStart = &(childStart[i]);
		t.minX = minX[i];
		t.maxX = maxX[i];
		t.barycenter = barycenter[i];
		t.radius = &(radius[i]);
		return t;
	}
	
	UNIVERSAL_STORAGE inline NodeArray<DIM, T> operator +(size_t i) const {
		NodeArray<DIM, T> o;
		o.isLeaf = isLeaf + i;
		o.childCount = childCount + i;
		o.childStart = childStart + i;
		o.minX = minX + i;
		o.maxX = maxX + i;
		o.barycenter = barycenter + i;
		o.mass = mass + i;
		o.radius = radius + i;
		return o;
	}
	
	
};

template<size_t DIM, typename T> struct Particle{
	Vec<DIM, T> pos;
	Vec<DIM, T> vel;
	T m;
};

template<size_t DIM, typename T> struct _ArrayParticleProxy{
	_ArrayVecProxy<DIM, T> pos;
	_ArrayVecProxy<DIM, T> vel;
	T *m;
	
	UNIVERSAL_STORAGE inline _ArrayParticleProxy<DIM, T>& operator=(const Particle<DIM, T> &v) {
		pos = v.pos;
		vel = v.vel;
		*m = v.ma;
		return *this;
	}
};

template<size_t DIM, typename T> struct ParticleArray{
	VecArray<DIM, T> pos;
	VecArray<DIM, T> vel;
	T *m;
	
	UNIVERSAL_STORAGE inline Particle<DIM, T> operator [](size_t i) const {
		Particle<DIM, T> t;
		t.pos = pos[i];
		t.vel = vel[i];
		t.m = m[i];
		
		return t;
	}
	
	
	UNIVERSAL_STORAGE inline _ArrayParticleProxy<DIM, T> operator [](size_t i) {
		_ArrayParticleProxy<DIM, T> t;
		t.pos = pos[i];
		t.vel = vel[i];
		t.m = &(m[i]);
		return t;
	}
	
	UNIVERSAL_STORAGE inline ParticleArray<DIM, T> operator +(size_t i) const {
		ParticleArray<DIM, T> o;
		o.pos = pos + i;
		o.vel = vel + i;
		o.m = m + i;
		return o;
	}
	
};



enum TraverseMode {
	CountOnly,
	HashInteractions,
	Forces
};

//#ifndef _COMPILE_FOR_CUDA_
// Double the metaprogramming techniques, double the fun
constexpr size_t InteractionElems(TraverseMode Mode, size_t DIM){
	return (Mode == CountOnly || Mode == HashInteractions) ? 2 : DIM;
}

template <size_t DIM, typename Float, TraverseMode Mode>
using  InteractionType = Vec<InteractionElems(Mode, DIM) , typename std::conditional<Mode == CountOnly || Mode == HashInteractions, size_t, Float>::type >;

template <size_t DIM, typename Float, TraverseMode Mode>
using  InteractionTypeArray = VecArray<InteractionElems(Mode, DIM) , typename std::conditional<Mode == CountOnly || Mode == HashInteractions, size_t, Float>::type >;

//#endif

/*

template<size_t DIM, typename T> struct ParticleComparator{
	Vec<DIM, T> minX;
	Vec<DIM, T> maxX;
 bool operator()(Particle<DIM, T> p1, Particle<DIM, T> p2){
	 return zIndex<DIM, T, unsigned char, size_t>(p1.pos, minX, maxX) <
	 zIndex<DIM, T, unsigned char, size_t>(p2.pos, minX, maxX);
 }
};
*/

#endif