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
    gl_FragColor = brdf * vec4(1.0, 1.0, 0.0, 1.0);

}











