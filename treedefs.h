#ifndef _TreeTraverse_TreeDefs_h_
#define _TreeTraverse_TreeDefs_h_

#ifdef __CUDA_ARCH__
#include <cuda.h>
#include <device_functions.h>
#define UNIVERSAL_STORAGE __host__ __device__
#else
#include <cstddef>
#include <cmath>
#include <cstdio>
#define UNIVERSAL_STORAGE
#endif

#include <type_traits>

typedef unsigned short uint16;

#define ASSERT_ARRAY_BOUNDS(i, elems) if(i >= elems){ \
	printf("%s @ %s:%d: Out of bounds access: %lu >= %lu\n",__func__, __FILE__, __LINE__,i,elems); \
}


template<size_t DIM, typename T> struct Vec{
	T x[DIM];

	UNIVERSAL_STORAGE Vec<DIM, T>(){}
	UNIVERSAL_STORAGE Vec<DIM, T>(const T a[DIM]){
		for(size_t i = 0; i < DIM; i++){
			x[i] = a[i];
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

template <size_t DIM, typename T> struct VecArray{
	T *x[DIM];
	size_t elems;
	
	UNIVERSAL_STORAGE VecArray<DIM, T>(){
		setCapacity(0);
	}
	UNIVERSAL_STORAGE VecArray<DIM, T>(Vec<DIM, T>& n){
		for(size_t i = 0; i < DIM; i++){
			x[i] = n.x+i;
		}
		setCapacity(1);
	}
	
	UNIVERSAL_STORAGE inline void setCapacity(size_t i){
		elems = i;
	}
	
	UNIVERSAL_STORAGE inline void get(size_t i, Vec<DIM, T> &t) const{
		ASSERT_ARRAY_BOUNDS(i, elems);
		for(size_t j = 0; j < DIM; j++){
			t.x[j] = x[j][i];
		}
	}
	
	
	UNIVERSAL_STORAGE inline void set(size_t i, const Vec<DIM, T> &t) {
		ASSERT_ARRAY_BOUNDS(i, elems);
		for(size_t j = 0; j < DIM; j++){
			x[j][i] = t.x[j];
		}
	}
	
	UNIVERSAL_STORAGE inline VecArray<DIM, T> operator +(size_t i) const {
		ASSERT_ARRAY_BOUNDS(i, elems);
		
		VecArray<DIM, T> o;
		for(size_t j = 0; j < DIM; j++){
			o.x[j] = x[j] + i;
		}
		o.setCapacity(elems - i);
		return o;
	}
};

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

template<size_t DIM, typename T, size_t MAX_PARTS> struct GroupInfoArray{
	size_t *childCount;
	size_t *childStart;
	VecArray<DIM, T> minX;
	VecArray<DIM, T> maxX;
	VecArray<DIM, T> center;
	T *radius;
	size_t elems; // We should suppress elems generation in production code, which means macroing up, or somesuch

	UNIVERSAL_STORAGE GroupInfoArray<DIM, T, MAX_PARTS >() {
		setCapacity(0);
		childCount = nullptr;
		childStart = nullptr;
		radius = nullptr;
	}
	UNIVERSAL_STORAGE GroupInfoArray<DIM, T, MAX_PARTS>(GroupInfo<DIM, T, MAX_PARTS>& g){
		childCount = &(g.childCount);
		childStart = &(g.childStart);
		
		minX = VecArray<DIM, T>(g.minX);
		maxX = VecArray<DIM, T>(g.maxX);
		center = VecArray<DIM, T>(g.center);
		radius = &(g.radius);
		setCapacity(1);
	}
	
	
	UNIVERSAL_STORAGE inline void setCapacity(size_t i){
		elems = i;
		minX.setCapacity(i);
		maxX.setCapacity(i);
		center.setCapacity(i);
	}
	
	UNIVERSAL_STORAGE inline void get(size_t i, GroupInfo<DIM, T, MAX_PARTS > &t) const {
		ASSERT_ARRAY_BOUNDS(i, elems);
		t.childCount = childCount[i];
		t.childStart = childStart[i];
		minX.get(i, t.minX);
		maxX.get(i, t.maxX);
		center.get(i, t.center);
		t.radius = radius[i];
	}
	
	
	UNIVERSAL_STORAGE inline void set(size_t i, const GroupInfo<DIM, T, MAX_PARTS > &t) {
		ASSERT_ARRAY_BOUNDS(i, elems);
		childCount[i] = t.childCount;
		childStart[i] = t.childStart;
		minX.set(i, t.minX);
		maxX.set(i, t.maxX);
		center.set(i, t.center);
		radius[i] = t.radius;
	}
	
	UNIVERSAL_STORAGE inline GroupInfoArray<DIM, T, MAX_PARTS> operator +(size_t i) const {
		ASSERT_ARRAY_BOUNDS(i, elems);
		GroupInfoArray<DIM, T, MAX_PARTS> o;
		o.childCount = childCount + i;
		o.childStart = childStart + i;
		o.minX = minX + i;
		o.maxX = maxX + i;
		o.center = center + i;
		o.radius = radius + i;
		o.setCapacity(elems - i);
		return o;
	}
	
	
};

template<size_t DIM, typename T> struct PointMass{
	Vec<DIM, T> pos;
	T m;
};

template<size_t DIM, typename T> struct PointMassArray{
	VecArray<DIM, T> pos;
	T *m;
	size_t elems;
	
	UNIVERSAL_STORAGE PointMassArray<DIM, T>(){
		setCapacity(0);
		m = nullptr;
	}
	
	UNIVERSAL_STORAGE PointMassArray<DIM, T>(PointMass<DIM, T>& p){
		m = &(p.m);
		pos = VecArray<DIM, T>(p.pos);
		setCapacity(1);
	}
	
	
	UNIVERSAL_STORAGE inline void setCapacity(size_t i){
		elems = i;
		pos.setCapacity(i);
	}
	
	UNIVERSAL_STORAGE inline void get(size_t i, PointMass<DIM, T> &t) const {
		ASSERT_ARRAY_BOUNDS(i, elems)
		pos.get(i,t.pos);
		t.m = m[i];
	}
	
	
	UNIVERSAL_STORAGE inline void set(size_t i, const PointMass<DIM, T> &t) {
		ASSERT_ARRAY_BOUNDS(i, elems)
		pos.set(i, t.pos);
		m[i] = t.m;
	}
	
	UNIVERSAL_STORAGE inline PointMassArray<DIM, T> operator +(size_t i) const {
		ASSERT_ARRAY_BOUNDS(i, elems)
		PointMassArray<DIM, T> o;
		o.pos = pos + i;
		o.m = m + i;
		o.setCapacity(elems - i);
		return o;
	}
	
};




template<size_t DIM, typename T> struct Node{
	bool isLeaf;
	size_t childCount;
	size_t childStart;
	Vec<DIM, T> minX;
	Vec<DIM, T> maxX;
	PointMass<DIM, T> barycenter;
	T radius;
	
	UNIVERSAL_STORAGE Node<DIM, T>(){}
};

template<size_t DIM, typename T> struct NodeArray{
	bool *isLeaf;
	size_t *childCount;
	size_t *childStart;
	VecArray<DIM, T> minX;
	VecArray<DIM, T> maxX;
	PointMassArray<DIM, T> barycenter;
	T *radius;
	size_t elems;
	
	UNIVERSAL_STORAGE NodeArray<DIM, T>(){
		setCapacity(0);
		isLeaf = nullptr;
		childCount = nullptr;
		childStart = nullptr;
		radius = nullptr;
	}
	UNIVERSAL_STORAGE NodeArray<DIM, T>(Node<DIM, T>& n){
		isLeaf = &(n.isLeaf);
		childCount = &(n.childCount);
		childStart = &(n.childStart);
		
		minX = VecArray<DIM, T>(n.minX);
		maxX = VecArray<DIM, T>(n.maxX);
		barycenter = PointMassArray<DIM, T>(n.barycenter);
		
		radius = &(n.radius);
		setCapacity(1);
	}
	
	
	UNIVERSAL_STORAGE inline void setCapacity(size_t i){
		elems = i;
		minX.setCapacity(i);
		maxX.setCapacity(i);
		barycenter.setCapacity(i);
	}
	
	UNIVERSAL_STORAGE inline void get(size_t i, Node<DIM, T> & t) const {
		ASSERT_ARRAY_BOUNDS(i, elems)
		t.isLeaf = isLeaf[i];
		t.childCount = childCount[i];
		t.childStart = childStart[i];
		minX.get(i,t.minX);
		maxX.get(i,t.maxX);
		barycenter.get(i,t.barycenter);
		t.radius = radius[i];
	}
	
	
	UNIVERSAL_STORAGE inline void set(size_t i, const  Node<DIM, T> & t) {
		ASSERT_ARRAY_BOUNDS(i, elems)
		isLeaf[i] = t.isLeaf;
		childCount[i] =t.childCount;
		childStart[i] = t.childStart;
		minX.set(i,t.minX);
		maxX.set(i,t.maxX);
		barycenter.set(i,t.barycenter);
		radius[i] = t.radius;
	}
	
	UNIVERSAL_STORAGE inline NodeArray<DIM, T> operator +(size_t i) const {
		ASSERT_ARRAY_BOUNDS(i, elems)
		NodeArray<DIM, T> o;
		o.isLeaf = isLeaf + i;
		o.childCount = childCount + i;
		o.childStart = childStart + i;
		o.minX = minX + i;
		o.maxX = maxX + i;
		o.barycenter = barycenter + i;
		o.radius = radius + i;
		o.setCapacity(elems - i);
		return o;
	}
	
	
};

template<size_t DIM, typename T> struct Particle{
	PointMass<DIM, T> mass;
	Vec<DIM, T> vel;
};

template<size_t DIM, typename T> struct ParticleArray{
	PointMassArray<DIM, T> mass;
	VecArray<DIM, T> vel;
	size_t elems;
	
	UNIVERSAL_STORAGE ParticleArray<DIM, T>(){
		setCapacity(0);
	}
	
	UNIVERSAL_STORAGE ParticleArray<DIM, T>(Particle<DIM, T>& p){
		mass = PointMassArray<DIM, T>(p.mass);
		vel = VecArray<DIM, T>(p.vel);
		setCapacity(1);
	}
	
	
	UNIVERSAL_STORAGE inline void setCapacity(size_t i){
		elems = i;
		mass.setCapacity(i);
		vel.setCapacity(i);
	}
	
	UNIVERSAL_STORAGE inline void get(size_t i, Particle<DIM, T> &t) const {
		ASSERT_ARRAY_BOUNDS(i, elems)
		mass.get(i,t.mass);
		vel.get(i,t.vel);
	}
	
	
	UNIVERSAL_STORAGE inline void set(size_t i, const Particle<DIM, T> &t) {
		ASSERT_ARRAY_BOUNDS(i, elems)
		mass.set(i, t.mass);
		vel.set(i, t.vel);
	}
	
	UNIVERSAL_STORAGE inline ParticleArray<DIM, T> operator +(size_t i) const {
		ASSERT_ARRAY_BOUNDS(i, elems)
		ParticleArray<DIM, T> o;
		o.mass = mass + i;
		o.vel = vel + i;
		o.setCapacity(elems - i);
		return o;
	}
	
};


enum TraverseMode {
	CountOnly,
	HashInteractions,
	Forces
};

constexpr UNIVERSAL_STORAGE bool NonForceCondition(TraverseMode Mode){
	return (Mode == CountOnly || Mode == HashInteractions);
}
// Double the metaprogramming techniques, double the fun
constexpr UNIVERSAL_STORAGE size_t InteractionElems(TraverseMode Mode, size_t DIM, size_t nonForceVal){
	return (NonForceCondition(Mode)) ? nonForceVal : DIM;
}


// Using crappy macros for this because template expansion is annoying
// We can make a nullary-template type, via closure with a struct typedef,
// or a ternary-template type, directly, but apparently no way to expose it as a template of two co-dependent arguments
#define InteractionType(DIM, Float, Mode) Vec<InteractionElems(Mode, DIM, 2) , typename std::conditional<NonForceCondition(Mode), size_t, Float>::type >
#define InteractionTypeArray(DIM, Float, Mode) VecArray<InteractionElems(Mode, DIM, 2) , typename std::conditional<NonForceCondition(Mode), size_t, Float>::type >
#define InteracterType(DIM, Float, Mode) PointMass<InteractionElems(Mode, DIM, 3), typename std::conditional<NonForceCondition(Mode), size_t, Float>::type >
#define InteracterTypeArray(DIM, Float, Mode) PointMassArray<InteractionElems(Mode, DIM, 3), typename std::conditional<NonForceCondition(Mode), size_t, Float>::type >

#endif
