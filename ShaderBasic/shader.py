# coding: utf-8
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


##############################################################################
# Shader
##############################################################################
# Checks for GL posted errors after appropriate calls
def printOpenGLError():
    err = glGetError()
    if (err != GL_NO_ERROR):
        print('GLERROR: ', gluErrorString(err))
        # sys.exit()


class Shader(object):
    def __init__(self):
        self.initShader('''

varying vec3 N;
varying vec3 v;
void main(void)  
{     
   v = vec3(gl_ModelViewMatrix * gl_Vertex);       
   N = normalize(gl_NormalMatrix * gl_Normal);
   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;  
}
        ''',
                        '''
                
                varying vec3 N;
                varying vec3 v;    
                void main (void)  
                {  
                   vec3 L = normalize(gl_LightSource[0].position.xyz - v);   
                   vec3 E = normalize(-v); // we are in Eye Coordinates, so EyePos is (0,0,0)  
                   vec3 R = normalize(-reflect(L,N));  
                
                   //calculate Ambient Term:  
                   vec4 Iamb = gl_FrontLightProduct[0].ambient;    
                
                   //calculate Diffuse Term:  
                   vec4 Idiff = gl_FrontLightProduct[0].diffuse * max(dot(N,L), 0.0);
                   Idiff = clamp(Idiff, 0.0, 1.0);     
                
                   // calculate Specular Term:
                   vec4 Ispec = gl_FrontLightProduct[0].specular 
                                * pow(max(dot(R,E),0.0),30.3*gl_FrontMaterial.shininess);
                   Ispec = clamp(Ispec, 0.0, 1.0); 
                   // write Total Color:  
                   gl_FragColor = gl_FrontLightModelProduct.sceneColor + Iamb + Idiff + Ispec;     
                }
                        ''')

    def initShader(self, vertex_shader_source, fragment_shader_source):
        # create program
        self.program = glCreateProgram()
        print('create program')
        printOpenGLError()

        # vertex shader
        print('compile vertex shader...')
        self.vs = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(self.vs, [vertex_shader_source])
        glCompileShader(self.vs)
        glAttachShader(self.program, self.vs)
        printOpenGLError()

        # fragment shader
        print('compile fragment shader...')
        self.fs = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(self.fs, [fragment_shader_source])
        glCompileShader(self.fs)
        glAttachShader(self.program, self.fs)
        printOpenGLError()

        print('link...')
        glLinkProgram(self.program)
        printOpenGLError()

    def begin(self):
        if glUseProgram(self.program):
            printOpenGLError()

    def end(self):
        glUseProgram(0)