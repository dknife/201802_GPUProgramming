//attribute vec3 Tangent
//attribute vec3 Binormal

varying out vec3 T;
varying out vec3 B;
varying out vec3 N;

varying out vec3 V;

void main() {
   N = normalize(gl_NormalMatrix * gl_Normal);
   T = Tangent;
   B = Tangent;
   V = vec3(gl_ModelViewMatrix * gl_Vertex);
   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
   gl_TexCoord[0] = gl_MultiTexCoord0;
}