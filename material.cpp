#include <vector>
#include "world.h"
#include "material.h"
#include "cudaHeaders.h"
#include <iostream>
#include <cmath>
#include <random>

unsigned int x=123456789, y=362436069, z=521288629;
__host__ __device__ float xorshf96() {          //3.6 times better performance than rand()
//	return 1.f*rand()/RAND_MAX;
	unsigned int t;
	x ^= x << 16;
	x ^= x >> 5;
	x ^= x << 1;
	t = x;
	x = y;
	y = z;
	z = t ^ x ^ y;
	return 1.0f*z/0xffffffff;
}

__host__ __device__ Color Material::shade(const Ray& incident, const bool isSolid) const
{
	if(!incident.didHit()) return world->getBackground();

	if(incident.intersected()->isLightSource() &&
	   xorshf96()<incident.intersected()->getLightSource()->getIntensity().maxComponent())  //randomly determine to emit light
		return incident.intersected()->getLightSource()->getIntensity();

	Color finalColor(color);

	double cosTheta = dotProduct(incident.getDirection(), incident.getNormal());
	Vector3D rDirection;
	rDirection = incident.getDirection() - 2 * incident.getNormal() * cosTheta;
	rDirection.normalize();
	Ray reflectedRay = Ray(incident.getPosition(), rDirection, incident.getLevel() + 1);  //Ideal reflection

	if(kt>0){
		//material is dielectric
		bool isInside = cosTheta > 0;
		double nc=1, nnt=isInside?eta/nc:nc/eta;
		double cos2t = 1-nnt*nnt*(1-cosTheta*cosTheta);
		if (cos2t<0){ //TIR
			return finalColor*world->shade_ray(reflectedRay);
		}else{
//			std::cout<<"asdasd"<<std::endl;
			cosTheta = -fabs(cosTheta);
			Vector3D tDirection = (incident.getDirection()*nnt - incident.getNormal()*((isInside?-1:1)*(cosTheta*nnt+sqrt(cos2t))));
			Ray refractedRay = Ray(incident.getPosition(), tDirection, incident.getLevel()+1);

			double a=eta-nc, b=eta+nc, R0=a*a/(b*b), c = 1-(isInside?dotProduct(tDirection, incident.getNormal()):-cosTheta);
			double Re=R0+(1-R0)*c*c*c*c*c,Tr=1-Re,P=.25+.5*Re,RP=Re/P,TP=Tr/(1-P);

			if(incident.getLevel()>1)
				if (xorshf96() < P) return finalColor * world->shade_ray(reflectedRay) * RP;
				else return finalColor * world->shade_ray(refractedRay) * TP;
			return finalColor * (world->shade_ray(reflectedRay)*Re + world->shade_ray(refractedRay)*Tr);
		}
	}else if(n>0)
	{
		double phi=2*M_PI*xorshf96(),
				cosAlpha=pow(xorshf96(), 1.f/(n+1)), sineAlpha = sqrt(1-cosAlpha*cosAlpha);
		double rotAngle = 2*(acos(-cosTheta) + acos(cosAlpha) - M_PI/2);

		Vector3D w=rDirection;
		Vector3D u=(crossProduct((fabs(w.X())>.1?Vector3D(0,1,0):Vector3D(1,0,0)),w));
		u.normalize();
		Vector3D v=crossProduct(w,u);

		Vector3D dDirection = u*cos(phi)*sineAlpha + v*sin(phi)*sineAlpha + w*cosAlpha;

		if(dotProduct(dDirection,incident.getNormal())<0) {
			Vector3D k = crossProduct(rDirection, incident.getNormal());
			k.normalize();
			dDirection = cos(rotAngle) * dDirection + sin(rotAngle) * crossProduct(k, dDirection);
		}
		Ray randomRay = Ray(incident.getPosition(), dDirection, incident.getLevel() + 1);  //Ideal reflection
		return finalColor*world->shade_ray(randomRay);
	}
	else if(!kr || xorshf96()>kr){ //kr=1 means completely reflective
		//do diffuse, Lambertian sampling
		double alpha=2*M_PI*xorshf96(),
				z=xorshf96(), sineTheta = sqrt(1-z);

		//generate basis
		Vector3D w=incident.getNormal();
		Vector3D u=(crossProduct((fabs(w.X())>.1?Vector3D(0,1,0):Vector3D(1,0,0)),w));
		u.normalize();
		Vector3D v=crossProduct(w,u);
		Vector3D dDirection = u*cos(alpha)*sineTheta + v*sin(alpha)*sineTheta + w*sqrt(z) ;
		Ray randomRay = Ray(incident.getPosition(), dDirection, incident.getLevel() + 1);  //Ideal reflection
		if(kg>0)
			return finalColor*world->shade_ray(randomRay) *(kg * pow(dotProduct(rDirection, dDirection), n)) * dotProduct(w, dDirection); //Glossy
		else
			return finalColor*world->shade_ray(randomRay);
	}else{
		//do specular
		return finalColor*world->shade_ray(reflectedRay);
	}
	return finalColor;

}
