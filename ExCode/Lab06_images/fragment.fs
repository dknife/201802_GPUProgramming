uniform sampler2D myTexture;
void main() {
    gl_FragColor = texture2D(myTexture, gl_TexCoord[0].yx);
}