uniform sampler2D myTexture;
uniform int imgW, imgH;

const int MAXKERNELSIZE = 25;
int kernelsize;
vec2 offset[MAXKERNELSIZE];
float filter[MAXKERNELSIZE];

void prepare33OffsetMatrix(void) {
    kernelsize = 9;

    float S = 1.0/float(imgW);
    float T = 1.0/float(imgH);

    offset[0]=vec2(-S, -T);
    offset[1]=vec2( 0, -T);
    offset[2]=vec2( S, -T);
    offset[3]=vec2(-S,  0);
    offset[4]=vec2( 0,  0);
    offset[5]=vec2( S,  0);
    offset[6]=vec2(-S,  T);
    offset[7]=vec2( 0,  T);
    offset[8]=vec2( S,  T);

}

void applyKernel(int size) {
    vec4 col = vec4(0.0);
    for (int i=0; i<size; i++) {
        col += texture2D(myTexture, gl_TexCoord[0].xy+offset[i])*(1.0/9.0);
    }
    gl_FragColor = col;
}
void main() {
    prepare33OffsetMatrix();
    applyKernel(9);
}



