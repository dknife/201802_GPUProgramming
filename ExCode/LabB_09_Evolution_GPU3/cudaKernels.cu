__device__ float fracf(float x) {
    return x - floorf(x);
}
__device__ float random (float s, float t, float *rSeed) {
    return fracf(sinf(s*12.98123198*rSeed[0] + t*78.231233*rSeed[1])*43758.5453123);
}

__device__ float fitness(
    float p1x, float p1y, float p2x, float p2y, float p3x, float p3y,
    float *obs, int nObs) {

    float ox, oy, ux, uy, vx, vy, wx, wy, pux, puy, pvx, pvy, pwx, pwy, cross1, cross2, cross3, fit;

    ux = p2x-p1x; uy = p2y-p1y;
    vx = p3x-p1x; vy = p3y-p1y;
    fit = ux*vy - uy*vx;
    if(fit<0) fit = -fit;

    for (int i=0; i<nObs; i++) {
        ox = obs[i*2+0];
        oy = obs[i*2+1];
        ux = p2x-p1x; uy=p2y-p1y;
        pux = ox-p1x; puy=oy-p1y;
        vx = p3x-p2x; vy=p3y-p2y;
        pvx = ox-p2x; pvy=oy-p2y;
        wx = p1x-p3x; wy=p1y-p3y;
        pwx = ox-p3x; pwy=oy-p3y;
        cross1 = ux*puy-uy*pux;
        cross2 = vx*pvy-vy*pvx;
        cross3 = wx*pwy-wy*pwx;
        if( cross1 > 0 && cross2 > 0 && cross3 > 0){ fit /= 2.0; }
        if( cross1 < 0 && cross2 < 0 && cross3 < 0){ fit /= 2.0; }
    }

    if(p1x > 1 || p1x < -1) { fit = 0; }
    if(p1y > 1 || p1y < -1) { fit = 0; }
    if(p2x > 1 || p2x < -1) { fit = 0; }
    if(p2y > 1 || p2y < -1) { fit = 0; }
    if(p3x > 1 || p3x < -1) { fit = 0; }
    if(p3y > 1 || p3y < -1) { fit = 0; }

    return fit;
}

__global__ void computeFitness(float *obs, float *gene, float *fit, int* metaData)
{
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx>metaData[1]) return;

    fit[idx] = fitness(
        gene[idx*6], gene[idx*6+1], gene[idx*6+2], gene[idx*6+3], gene[idx*6+4], gene[idx*6+5],
        obs, metaData[0]);

}

__global__ void rearrangePopulation(float *gene, float *fit, int* metaData)
{
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int nGene = metaData[1];
    int nHalf = nGene / 2;
    if(idx> nHalf) return;

    int j = nGene - 1 - idx;

    if (fit[idx] < fit[j]) {
        for(int k=0; k<6; k++) {
            float t = gene[idx*6+k];
            gene[idx*6+k] = gene[j*6+k];
            gene[j*6+k] = t;
            t = fit[idx];
            fit[idx] = fit[j];
            fit[j] = t;
        }
    }
}

__global__ void crossOver(float *gene, float *rSeed, int* metaData)
{
    const int idx = 2 * (threadIdx.x + blockDim.x*blockIdx.x);
    int nGene = metaData[1];
    int nHalf = nGene / 2;
    if(idx> nHalf) return;

    int childStart = int(nGene / 2);
    int j = idx + 1;
    int mutRegion = int(nHalf*1.5);
    int idx2 = idx + mutRegion;
    float w[6] = {
        random(gene[idx2*6], gene[j*6+1], rSeed),
        random(gene[idx2*6+1], gene[j*6], rSeed),
        random(gene[j*6], gene[idx2*6+1], rSeed),
        random(gene[idx2*6+1], gene[j*6+1], rSeed),
        random(gene[idx2*6], gene[j*6], rSeed),
        random(gene[j*6+1], gene[idx2*6+1], rSeed)
        };

    int childIdx = childStart + int(idx/2);
    for(int i=0; i<6; i++) {
        gene[childIdx*6 + i] = (1.0-w[i])*gene[idx*6+i] + w[i]*gene[j*6+i];
    }
}

__global__ void mutate(float *gene, float *rSeed, int* metaData)
{
    const int idx = (threadIdx.x + blockDim.x*blockIdx.x);
    int nGene = metaData[1];
    int nQuater = int(nGene / 4);
    if(idx> nQuater) return;

    int mutStart = int(nGene / 2) + nQuater;
    int i = mutStart + idx;

    float mut1 = 1.0 + (random(gene[i*6], gene[i*6], rSeed) - 0.5);
    float mut2 = 1.0 + (random(gene[i*6+1], gene[i*6], rSeed) - 0.5);
    float mut3 = 1.0 + (random(gene[i*6], gene[i*6+1], rSeed) - 0.5);

    gene[i*6+0] = mut1 * gene[idx*6+0];
    gene[i*6+1] = mut1 * gene[idx*6+1];
    gene[i*6+2] = mut2 * gene[idx*6+2];
    gene[i*6+3] = mut2 * gene[idx*6+3];
    gene[i*6+4] = mut3 * gene[idx*6+4];
    gene[i*6+5] = mut3 * gene[idx*6+5];

}

__global__ void shuffleGene(float *gene, float *fit, float *rSeed, int* metaData) {
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int nGene = metaData[1];
    int nHalf = nGene / 2;
    if(idx> nHalf) return;

    int Offset = int(nHalf/5.3);
    int j = nHalf + (idx + Offset)%nHalf;

    for(int k=0; k<6; k++) {
        float t = gene[idx*6+k];
        gene[idx*6+k] = gene[j*6+k];
        gene[j*6+k] = t;
        t = fit[idx];
        fit[idx] = fit[j];
        fit[j] = t;
    }
}