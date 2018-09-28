varying in vec3 N;
varying in vec3 V;
void main (void)
{
   N = normalize(N);
   vec3 L = normalize(gl_LightSource[0].position.xyz - V);
   vec3 E = normalize(-V);
   vec3 R = normalize(-reflect(L,N));


   //calculate Ambient Term:
   vec4 Iamb = gl_FrontLightProduct[0].ambient;

   //calculate Diffuse Term:
   vec4 Idiff = gl_FrontLightProduct[0].diffuse * max(dot(N,L), 0.0);
   Idiff = clamp(Idiff, 0.0, 1.0);

   // calculate Specular Term:
   vec4 Ispec = gl_FrontLightProduct[0].specular
                * pow(max(dot(R,E),0.0),gl_FrontMaterial.shininess);
   Ispec = clamp(Ispec, 0.0, 1.0);

   // write Total Color:
   gl_FragColor = Iamb + Idiff + Ispec;
}