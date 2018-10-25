varying vec3 V;

void main() {
   V = vec3(gl_ModelViewMatrix * gl_Vertex);
   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}