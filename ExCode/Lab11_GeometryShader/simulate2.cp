#version 320 es

uniform int nParticles;

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
vec3 gravity = vec3(0.0, -9.8, 0.0);
float e = 0.3;

float random (vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233)))*43758.5453123);
}

void main()
{
    uint idx = uint(3)*gl_GlobalInvocationID.x;
    if (idx > uint(nParticles*3) ) return;
    velOut[idx] = velIn[idx] + dt * gravity;
    locOut[idx] = locIn[idx] + dt * velOut[idx];
    vec3 l = locOut[idx];
    float d = sqrt(l.x*l.x+l.y*l.y+l.z*l.z);


    float penetration = 0.25-d;
    if ( penetration > 0.0) {
        vec3 N = l / d;
        locOut[idx] = N*(0.25+penetration);
        float dotNV = dot(N, velOut[idx]);
        if ( dotNV < 0.0 ) {
           velOut[idx] = reflect(velOut[idx], N) + (1.0-e)*dotNV*N;
       }
    }


    if ( locOut[idx].x > -0.5 && locOut[idx].x < 0.5 &&
          locOut[idx].z > -0.5 && locOut[idx].z < 0.5 ) {
        if ( locOut[idx].y < 0.0 ) {
            locOut[idx].y = -locOut[idx].y;
            if (velOut[idx].y < 0.0) {
                velOut[idx].y = -e*velOut[idx].y;
            }
        }
    }
    else if (locOut[idx].y<-1.0) {
        locOut[idx] = vec3(0.0, 0.35, 0.0);
        velOut[idx] = vec3(velOut[idx].z * 0.5, random(velOut[idx].xy)-0.5, velOut[idx].x*0.5);
    }
}