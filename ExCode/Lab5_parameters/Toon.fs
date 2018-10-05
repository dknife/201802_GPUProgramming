varying in vec3 N;
varying in vec3 V;

const float PI = 3.141592;

void main (void)
{
    N = normalize(N);
    vec3 L = normalize(gl_LightSource[0].position.xyz - V);
    vec3 E = normalize(-V);
    vec3 R = normalize(-reflect(L,N));
    vec3 H = (N+L)/2.0;
    vec2 alpha = vec2(10.0, 10.0);
    float d = alpha.x*H.x*H.x + alpha.y*H.y*H.y;
    float brdf = min( 0.5*((1.0-d)*cos(2.0*PI*sqrt(d)) + 1.0),  1.0);
    if (brdf < 0.2 ) brdf = 0.2;
    else if (brdf < 0.4 ) brdf = 0.4;
    else if (brdf < 0.6 ) brdf = 0.6;
    else if (brdf < 0.8 ) brdf = 0.8;
    else brdf = 1.0;
    if (dot(N,E) < 0.4 ) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
    else {
        gl_FragColor = brdf * vec4(1.0, 1.0, 0.0, 1.0);
    }

}