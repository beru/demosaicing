
#include <cuda_runtime.h>
#include <stdint.h>
#include "Timer.h"
#include <stdio.h>

__device__ uint16_t* d_src;
__device__ uchar4* d_dst;
__device__ __constant__ int d_part_length;

__global__
void demosaic(
	const uint16_t* pSrc,
	size_t width,
	size_t height,
	uchar4* pDst
	)
{
	int i;
	i = gridDim.x;
	i = blockDim.x;
	i = blockIdx.x;
	i = threadIdx.x;

//	int total = gridDim.x * blockDim.x;
	if (((threadIdx.x + 1) * 16 - 1) >= width) {
		return;
	}

	size_t offset = threadIdx.x * 2;
	pSrc += offset;
	pDst += offset;
	pSrc += width * blockIdx.x;
	pDst += width * (blockIdx.x + 1) + 1;

	// g r g r g r
	// b g b g b g
	// g r g r g r
	// b g b g b g
	const int nShifts = 8;
	if (blockIdx.x == 0) {
		for (size_t y=0; y<height; y+=2) {
			const uint16_t* pUp = pSrc;
			const uint16_t* pMi = pSrc + width;
			const uint16_t* pLo = pSrc + 2 * width;
			uchar4* pDst0 = pDst;
			for (size_t x=0; x<16; x+=2) {
				uint32_t r0 = pUp[1] + pLo[1] + 1;
				pDst0->x = r0 >> (1 + nShifts);
				pDst0->y = pMi[1] >> nShifts;
				pDst0->z = (pMi[0] + pMi[2] + 1) >> (1 + nShifts);
				++pDst0;

				pDst0->x = (r0 + pUp[3] + pLo[3] + 1) >> (2 + nShifts);
				pDst0->y = (pMi[1] + pMi[3] + 1) >> (1 + nShifts);
				pDst0->z = pMi[2] >> nShifts;
				++pDst0;

				pUp += 16;
				pMi += 16;
				pLo += 16;
				pDst0 += 14;
			}
			pSrc += width * 2;
			pDst += width * 2;
		}
	}else {
		for (size_t y=1; y<height; y+=2) {
			const uint16_t* pUp = pSrc;
			const uint16_t* pMi = pSrc + width;
			const uint16_t* pLo = pSrc + 2 * width;
			uchar4* pDst0 = pDst;
			for (size_t x=0; x<16; x+=2) {
				pDst0->x = pMi[1] >> nShifts;
				pDst0->y = (pUp[1] + pMi[0] + pMi[2] + pLo[1] + 2) >> (2 + nShifts);
				pDst0->z = (pUp[0] + pUp[2] + pLo[0] + pLo[2] + 2) >> (2 + nShifts);
				++pDst0;

				pDst0->x = (pMi[1] + pMi[3] + 1) >> (1 + nShifts);
				pDst0->y = pMi[2] >> nShifts;
				pDst0->z = (pUp[2] + pLo[2] + 1) >> (1 + nShifts);
				++pDst0;

				pUp += 16;
				pMi += 16;
				pLo += 16;
				pDst0 += 14;
			}
			pSrc += width * 2;
			pDst += width * 2;
		}
	}

}

void cuda_demosaic_grbg(
	const uint16_t* pSrc,
	size_t width,
	size_t height,
	uint32_t* pDst
	)
{
	cudaError_t ret;
Timer t;

	uint32_t* h_dst;
	ret = cudaMallocHost((void**)&h_dst, width*height*sizeof(uchar4));
	ret = cudaMalloc((void**)&d_src, width*height*sizeof(ushort1));
	ret = cudaMalloc((void**)&d_dst, width*height*sizeof(uchar4));
printf("Elapsed %f\n", t.ElapsedSecond());
t.Start();
	ret = cudaMemcpy(d_src, pSrc, width*height*sizeof(ushort1), cudaMemcpyHostToDevice);
	ret = cudaMemcpy(d_dst, pDst, width*height*sizeof(uint32_t), cudaMemcpyHostToDevice);
printf("Elapsed %f\n", t.ElapsedSecond());
t.Start();

	int numBlocksInAGrid = 2;
	int numThreadsInABlock = 256;

	int len = (width / numThreadsInABlock + 1) & ~1;

	ret = cudaMemcpyToSymbol((const void*)&d_part_length, &len, sizeof(len));

t.Start();
	demosaic<<<numBlocksInAGrid,numThreadsInABlock>>>(d_src, width, height, d_dst);
	cudaThreadSynchronize();
printf("Elapsed %f\n", t.ElapsedSecond());
t.Start();
	//ret = cudaMemcpy(h_dst, d_dst, width*height*sizeof(uchar4), cudaMemcpyDeviceToHost);
	ret = cudaMemcpy(pDst, d_dst, width*height*sizeof(uchar4), cudaMemcpyDeviceToHost);
printf("Elapsed %f\n", t.ElapsedSecond());

	ret = cudaFree(d_src);
	ret = cudaFree(d_dst);
	ret = cudaFreeHost(h_dst);
}
