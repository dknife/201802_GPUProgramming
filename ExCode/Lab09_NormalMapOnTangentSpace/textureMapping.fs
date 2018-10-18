uniform sampler2D myTexture;

varying vec3 T;
varying vec3 B;
varying vec3 N;
varying vec3 V;

void main (void)
{


    vec3 Nmap = 2.0 *(texture2D(myTexture, gl_TexCoord[0].xy).xyz -0.5 );
    vec3 L = normalize(gl_LightSource[0].position.xyz - V);
    vec3 E = normalize(-V);
    L = vec3(dot(T,L), dot(B,L), dot(N,L));
    E = vec3(dot(T,E), dot(B,E), dot(N,E));
    vec3 N = Nmap;


    vec3 R = normalize(-reflect(L,N));

    //calculate Diffuse Intesity
    float Idiff = max(dot(N,L), 0.0);
    Idiff = clamp(Idiff, 0.0, 1.0);

    // calculate Specular Intesity
    float Ispec = pow(max(dot(R,E),0.0),gl_FrontMaterial.shininess);
    Ispec = clamp(Ispec, 0.0, 1.0);

    vec2 alpha = vec2(5.0, 5.0);
    vec3 H = normalize(N+L+E);
    float d = alpha.x*H.x*H.x + alpha.y*H.y*H.y;
    float brdf = min( 0.5*((1.0-d)*cos(2.0*3.14*sqrt(d)) + 1.0),  1.0);

    // write Total Color:
    gl_FragColor = (0.5*Idiff+1.0*Ispec)*vec4(1.0, 0.9, 0.5, 1.0) + 0.5*brdf*(Idiff+0.5)*vec4(1.0, 1.0, 0.5, 1.0);

}