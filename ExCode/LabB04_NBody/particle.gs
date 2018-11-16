#version 330 compatibility
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

out vec2 Vertex_UV;

float random (vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233)))*43758.5453123);
}

void main()
{

    //vec2 texOffset = vec2(0.1*random(gl_in[0].gl_Position.xy), 0.2*random(gl_in[0].gl_Position.yz));

    float d = 0.005;
    gl_Position = gl_in[0].gl_Position + vec4(-d, d, 0, 0);
    Vertex_UV = vec2(0.0, 0.0);//+texOffset;
    EmitVertex();
    gl_Position = gl_in[0].gl_Position + vec4(-d,-d, 0, 0);
    Vertex_UV = vec2(0.0, 1.0);//+texOffset;
    EmitVertex();
    gl_Position = gl_in[0].gl_Position + vec4( d, d, 0, 0);
    Vertex_UV = vec2(1.0, 0.0);//+texOffset;
    EmitVertex();
    gl_Position = gl_in[0].gl_Position + vec4( d,-d, 0, 0);
    Vertex_UV = vec2(1.0, 1.0);//+texOffset;
    EmitVertex();


    EndPrimitive();
}