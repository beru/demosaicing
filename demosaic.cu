
#include <cuda_runtime.h>
#include <stdint.h>
#include "Timer.h"
#include <stdio.h>

__device__ __constant__ int d_part_length;

__global__
void demosaic(
	const uint16_t* __restrict__ pSrc,
	uchar4* __restrict__ pDst,
	size_t width,
	size_t height,
	size_t src_pitch,
	size_t dst_pitch
	)
{
	int i;
	i = gridDim.x;
	i = blockDim.x;
	i = blockIdx.x;
	i = threadIdx.x;

	const size_t y_offset = blockIdx.x * 2;
	const size_t y_loop_count = (height - gridDim.x) / gridDim.x;
	const size_t y_stride = gridDim.x;

	volatile size_t index = threadIdx.x;
	size_t warpIndex = index / warpSize;
	index -= ((warpIndex + 1) >> 1) * warpSize;
	size_t x_offset = 512 * (index / 32u) + (index & 31u) * 2;
	const size_t x_step = 64u;
	const size_t x_loop_count = 8u;
	if (x_offset >= width) {
		return;
	}
	pSrc += x_offset;
	pDst += x_offset;

	pSrc = (const uint16_t*)((const char*)pSrc + y_offset * src_pitch);
	pDst = (uchar4*)((char*)pDst + y_offset * dst_pitch);
	// çsï™ÇØ
	pSrc = (const uint16_t*)((const char*)pSrc + (warpIndex & 1) * src_pitch);
	pDst = (uchar4*)((char*)pDst + ((warpIndex & 1) + 1) * dst_pitch) + 1;
	// g r g r g r
	// b g b g b g
	// g r g r g r
	// b g b g b g
	const int nShifts = 8;
	if (warpIndex & 1) {
		for (int y=0; y<y_loop_count; ++y) {
			const uint16_t* pUp = pSrc;
			const uint16_t* pMi = (const uint16_t*)((const char*)pSrc + src_pitch);
			const uint16_t* pLo = (const uint16_t*)((const char*)pSrc + 2 * src_pitch);
			uchar4* pDst0 = pDst;
			for (int x=0; x<x_loop_count; ++x) {
				pDst0->x = pMi[1] >> nShifts;
				pDst0->y = (pUp[1] + pMi[0] + pMi[2] + pLo[1] + 2) >> (2 + nShifts);
				pDst0->z = (pUp[0] + pUp[2] + pLo[0] + pLo[2] + 2) >> (2 + nShifts);
				++pDst0;

				pDst0->x = (pMi[1] + pMi[3] + 1) >> (1 + nShifts);
				pDst0->y = pMi[2] >> nShifts;
				pDst0->z = (pUp[2] + pLo[2] + 1) >> (1 + nShifts);
				++pDst0;

				pUp += x_step;
				pMi += x_step;
				pLo += x_step;
				pDst0 += x_step - 2;
			}
			pSrc = (const uint16_t*)((const char*)pSrc + y_stride * src_pitch);
			pDst = (uchar4*)((char*)pDst + y_stride * dst_pitch);
		}
	}else {
		for (int y=0; y<y_loop_count; ++y) {
			const uint16_t* pUp = pSrc;
			const uint16_t* pMi = (const uint16_t*)((const char*)pSrc + src_pitch);
			const uint16_t* pLo = (const uint16_t*)((const char*)pSrc + 2 * src_pitch);
			uchar4* pDst0 = pDst;
			for (int x=0; x<x_loop_count; ++x) {
				uint32_t r0 = pUp[1] + pLo[1] + 1;
				pDst0->x = r0 >> (1 + nShifts);
				pDst0->y = pMi[1] >> nShifts;
				pDst0->z = (pMi[0] + pMi[2] + 1) >> (1 + nShifts);
				++pDst0;

				pDst0->x = (r0 + pUp[3] + pLo[3] + 1) >> (2 + nShifts);
				pDst0->y = (pMi[1] + pMi[3] + 1) >> (1 + nShifts);
				pDst0->z = pMi[2] >> nShifts;
				++pDst0;

				pUp += x_step;
				pMi += x_step;
				pLo += x_step;
				pDst0 += x_step - 2;
			}
			pSrc = (const uint16_t*)((const char*)pSrc + y_stride * src_pitch);
			pDst = (uchar4*)((char*)pDst + y_stride * dst_pitch);
		}
	}

}

void cuda_demosaic_grbg(
	const uint16_t* __restrict pSrc,
	size_t width,
	size_t height,
	uint32_t* __restrict pDst
	)
{
	cudaError_t ret;
Timer t;

	uint16_t* d_src;
	uint16_t* h_src;
	uchar4* d_dst;
	uchar4* h_dst;
	size_t d_src_pitch, d_dst_pitch;
	ret = cudaMallocPitch((void**)&d_src, &d_src_pitch, width*sizeof(uint16_t), height);
	ret = cudaMallocPitch((void**)&d_dst, &d_dst_pitch, width*sizeof(uchar4), height);
	ret = cudaMallocHost((void**)&h_src, d_src_pitch*height);
	ret = cudaMallocHost((void**)&h_dst, d_dst_pitch*height);
printf("cudaMalloc Elapsed %f\n", t.ElapsedSecond());
t.Start();
	for (size_t y=0; y<height; ++y) {
		memcpy((uint8_t*)h_src + y*d_src_pitch, pSrc + y*width, d_src_pitch);
	}
	ret = cudaMemcpy(d_src, h_src, d_src_pitch*height, cudaMemcpyHostToDevice);
//	ret = cudaMemcpy(d_dst, h_dst, dst_pitch*height, cudaMemcpyHostToDevice);
printf("cudaMemcpy Elapsed %f\n", t.ElapsedSecond());
t.Start();

	int numBlocksInAGrid = 64;
	int numThreadsInABlock = 512;

	int len = (width / numThreadsInABlock + 1) & ~1;

	ret = cudaMemcpyToSymbol((const void*)&d_part_length, &len, sizeof(len));

t.Start();
	demosaic<<<numBlocksInAGrid,numThreadsInABlock>>>(d_src, d_dst, width, height, d_src_pitch, d_dst_pitch);
	ret = cudaThreadSynchronize();
printf("demosaic Elapsed %f\n", t.ElapsedSecond());
t.Start();
	ret = cudaMemcpy(h_dst, d_dst, d_dst_pitch*height, cudaMemcpyDeviceToHost);
	for (size_t y=0; y<height; ++y) {
		memcpy(pDst + y*width, (uint8_t*)h_dst + y*d_dst_pitch, width * sizeof(uchar4));
	}

printf("cudaMemcpy Elapsed %f\n", t.ElapsedSecond());

	ret = cudaFree(d_src);
	ret = cudaFreeHost(h_src);
	ret = cudaFree(d_dst);
	ret = cudaFreeHost(h_dst);
}
