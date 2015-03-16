// We want to transform AOS into SOA wherever possible!
// This is something we can do with a macro =)

#include <cstddef>
#include <cmath>

typedef unsigned short uint16;

template<size_t DIM, typename T> struct Vec{
	T x[DIM];

	inline Vec<DIM, T> operator -(const Vec<DIM, T> &v) const{
		Vec<DIM, T> out;
		for(size_t i = 0; i < DIM; i++){
			out.x[i] = this->x[i] - v.x[i];
		}
		return out;
	}
	
	inline Vec<DIM, bool> operator <(const Vec<DIM, T> &v) const{
		Vec<DIM, bool> out;
		for(size_t i = 0; i < DIM; i++){
			out.x[i] = this->x[i] < v.x[i];
		}
		return out;
	}
	
	inline Vec<DIM, T> operator +(const Vec<DIM, T> &v) const{
		Vec<DIM, T> out;
		for(size_t i = 0; i < DIM; i++){
			out.x[i] = this->x[i] + v.x[i];
		}
		return out;
	}
	
	inline Vec<DIM, T> operator *(T s) const{
		Vec<DIM, T> out;
		for(size_t i = 0; i < DIM; i++){
			out.x[i] = this->x[i] * s;
		}
		return out;
	}
	
	inline Vec<DIM, T> operator /(T s) const{
		Vec<DIM, T> out;
		for(size_t i = 0; i < DIM; i++){
			out.x[i] = this->x[i] / s;
		}
		return out;
	}
	
	inline Vec<DIM, T> operator *(const Vec<DIM, T> &v) const{
		Vec<DIM, T> out;
		for(size_t i = 0; i < DIM; i++){
			out.x[i] = this->x[i] * v.x[i];
		}
		return out;
	}
	
	inline Vec<DIM, T> operator /(const Vec<DIM, T> &v) const{
		Vec<DIM, T> out;
		for(size_t i = 0; i < DIM; i++){
			out.x[i] = this->x[i] / v.x[i];
		}
		return out;
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


template<size_t DIM, typename T> bool contains(const Vec<DIM, T> &lower, const Vec<DIM, T> &upper, const Vec<DIM, T> point){
	bool is_contained = true;
	for(size_t i = 0; is_contained && i < DIM; i++){
		is_contained &= (lower.x[i] <= point.x[i]) && (upper.x[i] >= point.x[i]);
	}
	return is_contained;
}

template<size_t DIM, typename T> Vec<DIM, T> min(const Vec<DIM, T> &v1, const Vec<DIM, T> &v2){
	Vec<DIM, T> ret;
	for(size_t i = 0; i < DIM; i++){
		ret.x[i] = (v1.x[i] < v2.x[i]) ? v1.x[i] : v2.x[i];
	}
	return ret;
}


template<size_t DIM, typename T> Vec<DIM, T> max(const Vec<DIM, T> &v1, const Vec<DIM, T> &v2){
	Vec<DIM, T> ret;
	for(size_t i = 0; i < DIM; i++){
		ret.x[i] = (v1.x[i] > v2.x[i]) ? v1.x[i] : v2.x[i];
	}
	return ret;
}


template<size_t DIM, typename T> T mag_sq(const Vec<DIM, T> &v){
	T ret = 0;
	for(size_t i = 0; i < DIM; i++){
		ret += v.x[i] * v.x[i];
	}
	return ret;
}


template<size_t DIM, typename T> T mag(const Vec<DIM, T> &v){
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

template<size_t DIM, typename T> struct Node{
	bool isLeaf;
	size_t childCount;
	size_t childStart;
	Vec<DIM, T> minX;
	Vec<DIM, T> maxX;
	T mass;
	Vec<DIM, T> barycenter;
	T radius;
};

template<size_t DIM, typename T> struct Particle{
	T m;
	Vec<DIM, T> pos;
	Vec<DIM, T> vel;
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