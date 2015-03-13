#include "treedefs.h"
#include <iostream>

template<typename T> T factorial(T n)
{
  return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

template<typename Float> Float readFloat(FILE *f) {
  Float v;
  fread((void*)(&v), sizeof(v), 1, f);
  return v;
}


template<size_t DIM, typename Float> Float sys_energy(size_t n, Particle<DIM, Float> *particles){
	Float energy = 0.0;
	for(size_t i = 0; i < n; i++){
		energy += 0.5 * particles[i].m * mag_sq(particles[i].vel);
		for(size_t j = 0; j < i; j++){
			energy -= particles[i].m * particles[j].m / mag(particles[i].pos - particles[j].pos);
		}
	}
	return energy;
}

template<size_t ORDER, typename Float> Float forward_euler(Vec<ORDER+1, Float> yp, Float dt){
	Float yn = 0.0;
	for(size_t i = 0; i <= ORDER; i++){
		yn += (yp.x[i] / factorial(i)) * pow(dt, i);
	}
	return yn;
}

template<size_t DIM, typename Float> void integrate_system(size_t n, Particle<DIM, Float> *particles, Vec<DIM, Float> *forces, Float dt){
	for(size_t i = 0; i < n; i++){
		for(size_t j = 0; j < DIM; j++){
			Vec<3, Float> posp = {.x = {particles[i].pos.x[j], particles[i].vel.x[j], forces[i].x[j] / particles[i].m}};
			Vec<2, Float> velp = {.x = {particles[i].vel.x[j], forces[i].x[j] / particles[i].m}};
			particles[i].pos.x[j] = forward_euler<2, Float>(posp, dt);
			particles[i].vel.x[j] = forward_euler<1, Float>(velp, dt);	
		}
	}
	
}

template<size_t DIM, typename Float> void calc_forces_bruteforce(size_t n, Particle<DIM, Float> *particles, Vec<DIM, Float> *forces){
	for(size_t i = 0; i < n; i++){
		forces[i] = (Vec<DIM,Float>){.x = {0, 0, 0}};
		for(size_t j = 0; j< n; j++){
			if(i == j) j++;
			Vec<DIM, Float> disp = particles[i].pos - particles[j].pos;
			forces[i] = (forces[i] + disp * ((particles[i].m * particles[j].m) / (Float)pow(mag_sq(disp),1.5)));
		}
	}
}




#define DIM 3
#define Float float
#define DT 0.001


int main(int argc, char* argv[]) {
	int nPs = atoi(argv[1]);
	Particle<DIM,Float> *bodies = new Particle<DIM,Float>[nPs] ;
	Vec<DIM,Float> *forces = new Vec<DIM,Float>[nPs];
	FILE *f = fopen(argv[2],"rb");
	for(size_t i = 0; i < nPs; i++){
		bodies[i].m = readFloat<Float>(f);
		for(size_t j = 0; j < DIM; j++){
			bodies[i].pos.x[j] = readFloat<Float>(f);		
		}
		for(int j = 0; j < DIM; j++){
			bodies[i].vel.x[j] = readFloat<Float>(f);
		}
	}
	fclose(f);
	
	Float e_init = sys_energy<DIM, Float>(nPs, bodies);
	
	std::cout << nPs << "\t" << (1.0  / nPs) << "\t" << bodies[0].m << "\t" << bodies[nPs - 1].m << std::endl;
	std::cout << "Init energy:\t" << e_init << std::endl;
	for(int i = 0; i < 1000; i++){
		calc_forces_bruteforce<DIM, Float>(nPs, bodies, forces);
		integrate_system<DIM, Float>(nPs, bodies, forces, DT);
		Float e_now = sys_energy<DIM, Float>(nPs, bodies);
		std::cout << "%dE:\t" << (e_init - e_now) / e_init << std::endl;
	}
	
	
	
	delete [] bodies;
}
