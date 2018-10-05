uniform sampler2D myTexture;
void main() {
    gl_FragColor = 0.5*texture2D(myTexture, gl_TexCoord[0].xy);
}