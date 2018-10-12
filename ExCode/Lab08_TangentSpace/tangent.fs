uniform sampler2D myTexture;

varying in vec3 T;
varying in vec3 B;
varying in vec3 N;

varying in vec3 V;

void main (void)
{
    N = normalize(N);
    vec3 L = normalize(gl_LightSource[0].position.xyz - V);
    vec3 E = normalize(-V);
    vec3 R = normalize(-reflect(L,N));

    N = 2.0 *(texture2D(myTexture, gl_TexCoord[0].xy).xyz -0.5 );

    //calculate Diffuse Intesity
    float Idiff = max(dot(N,L), 0.0);
    Idiff = clamp(Idiff, 0.0, 1.0);

    // calculate Specular Intesity
    vec4 Ispec = pow(max(dot(R,E),0.0),gl_FrontMaterial.shininess);
    Ispec = clamp(Ispec, 0.0, 1.0);

    // write Total Color:
    gl_FragColor = (Idiff+Ispec)*vec4(1.0, 1.0, 0.5, 1.0);
}