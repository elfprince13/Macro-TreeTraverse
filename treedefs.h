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
};


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

template<size_t DIM, size_t MAX_PARTS> struct GroupInfo{
	size_t nParts;
	size_t partIds[MAX_PARTS];
	Vec<DIM, uint16> minX;
	Vec<DIM, uint16> maxX;
};

template<size_t DIM> struct Node{
	bool isLeaf;
	size_t childCount;
	size_t childStart;
	Vec<DIM, uint16> minX;
	Vec<DIM, uint16> maxX;
};

template<size_t DIM, typename T> struct Particle{
	T m;
	Vec<DIM, T> pos;
	Vec<DIM, T> vel;
};