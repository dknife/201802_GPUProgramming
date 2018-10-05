varying out vec3 N;
varying out vec3 V;
void main(void)
{
   N = normalize(gl_NormalMatrix * gl_Normal);
   V = vec3(gl_ModelViewMatrix * gl_Vertex);
   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}