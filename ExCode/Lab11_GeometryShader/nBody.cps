#version 320 es

uniform int nParticles;
uniform vec3 wind;

layout(local_size_x = 10, local_size_y=1, local_size_z=1) in;


layout(binding = 0)  buffer LocIn {
    vec3 locIn[];
};


layout(binding = 1)  buffer VelIn {
    vec3 velIn[];
};

layout(binding = 2)  buffer LocOut {
    vec3 locOut[];
};

layout(binding = 3)  buffer VelOut {
    vec3 velOut[];
};

float dt = 0.01;

vec3 gravity = vec3(0.0, -9.8, 0.0) ;
float e = 0.3;

float random (vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233)))*43758.5453123);
}

void main()
{

    gravity = gravity + wind;
    uint index = gl_GlobalInvocationID.x;
    if (index > uint(nParticles) ) return;

    uint vIdx = index*uint(3);

    vec3 force = vec3(0.0);
    for (int idx=0; idx<nParticles; idx++) {
        vec3 xji = locIn[uint(idx*3)] - locIn[vIdx];
        float lSqr = xji.x*xji.x + xji.y*xji.y + xji.z*xji.z ;
        float l = sqrt(lSqr);
        if(l>0.02) {
            force = force + xji/(lSqr*l);
        }
        else {
            vec3 temp = (velIn[vIdx]  + velIn[uint(idx*3)])/2.0;
            velIn[vIdx] = temp;
            velIn[uint(idx*3)] = temp;
        }
    }
    velOut[vIdx] = velIn[vIdx] + dt * 0.00005 * force;
    locOut[vIdx] = locIn[vIdx] + dt * velOut[vIdx];
    vec3 l = locOut[vIdx];
    float d = sqrt(l.x*l.x+l.y*l.y+l.z*l.z);




}