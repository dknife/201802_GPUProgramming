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

    gravity.y = -9.8*locIn[vIdx].y;

    velOut[vIdx] = velIn[vIdx] + dt * gravity;
    locOut[vIdx] = locIn[vIdx] + dt * velOut[vIdx];
    vec3 l = locOut[vIdx];
    float d = sqrt(l.x*l.x+l.y*l.y+l.z*l.z);


    float penetration = 0.25-d;
    if ( penetration > 0.0) {
        vec3 N = l / d;
        locOut[vIdx] = N*(0.25+penetration);
        float dotNV = dot(N, velOut[vIdx]);
        if ( dotNV < 0.0 ) {
           velOut[vIdx] = reflect(velOut[vIdx], N) + (1.0-e)*dotNV*N;
       }
    }


    if ( locOut[vIdx].x > -0.5 && locOut[vIdx].x < 0.5 &&
          locOut[vIdx].z > -0.5 && locOut[vIdx].z < 0.5 ) {
        if ( locOut[vIdx].y < 0.0 ) {
            locOut[vIdx].y = -locOut[vIdx].y;
            if (velOut[vIdx].y < 0.0) {
                velOut[vIdx].y = -e*velOut[vIdx].y;
            }
        }
    }
    else if (locOut[vIdx].y<-1.0 || d > 1.0) {
        locOut[vIdx] = vec3(0.0, 0.35, 0.0);
        //velOut[vIdx] = vec3(velOut[vIdx].z * 0.0, random(velOut[vIdx].xy)-0.5, velOut[vIdx].x*0.0);
        velOut[vIdx] = vec3(random(velOut[vIdx].yz)-0.5, random(velOut[vIdx].xy)-0.5, random(velOut[vIdx].xz)-0.5);
    }

}