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
    float Ispec = pow(max(dot(R,E),0.0),3.0); //gl_FrontMaterial.shininess);
    Ispec = clamp(Ispec, 0.0, 1.0);

    // write Total Color:
    gl_FragColor = vec4( (0.25*Idiff+0.5*Ispec)*vec3(1.0, 0.9, 0.5), 1.0);

}