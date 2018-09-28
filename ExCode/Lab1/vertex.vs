
uniform float time;

void main(void)
{
    gl_Vertex.y = 0.2*sin(time*(gl_Vertex.x+gl_Vertex.z));
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}