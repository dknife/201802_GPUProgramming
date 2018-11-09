attribute vec3 Tangent;
attribute vec3 Binormal;

varying vec3 T;
varying vec3 B;
varying vec3 N;
varying vec3 V;

void main() {
   T = normalize(gl_NormalMatrix * Tangent);
   B = normalize(gl_NormalMatrix * Binormal);
   N = normalize(gl_NormalMatrix * gl_Normal);

   V = vec3(gl_ModelViewMatrix * gl_Vertex);
   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
   gl_TexCoord[0] = gl_MultiTexCoord0;
}