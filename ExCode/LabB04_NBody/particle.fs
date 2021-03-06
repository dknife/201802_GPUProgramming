uniform sampler2D sprite;

in vec2 Vertex_UV;

void main (void)
{
    vec4 color = texture2D(sprite, Vertex_UV.xy);
    gl_FragColor = color*vec4( 0.7, 0.7, 1.0, 0.5*color.r*color.g*color.b);
}