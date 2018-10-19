#version 320 es

layout(local_size_x = 10) in;

layout(binding = 0) readonly buffer Input0 {
    float data[];
} input0;

layout(binding = 1) readonly buffer Input1 {
    float data[];
} input1;

layout(binding = 2) writeonly buffer Output {
    float data[];
} output0;


void main()
{
    uint idx = gl_GlobalInvocationID.x;
    output0.data[idx] = input0.data[idx] - input1.data[idx];
}