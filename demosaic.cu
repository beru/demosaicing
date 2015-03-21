
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "Timer.h"

#if 1
#define OFFSET_PTR(ptr, num_bytes) ((decltype(ptr))((const char*)ptr + num_bytes))
#else
template <typename T>
__device__
T* OFFSET_PTR(T* ptr, int num_bytes)
{
	return (T*)((char*)(ptr) + num_bytes);
}
#endif

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
	//int i;
	//i = gridDim.x;
	//i = blockDim.x;
	//i = blockIdx.x;
	//i = threadIdx.x;

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

	pSrc = OFFSET_PTR(pSrc, y_offset * src_pitch);
	pDst = OFFSET_PTR(pDst, y_offset * dst_pitch);
	// çsï™ÇØ
	pSrc = OFFSET_PTR(pSrc, (warpIndex & 1) * src_pitch);
	pDst = OFFSET_PTR(pDst, ((warpIndex & 1) + 1) * dst_pitch) + 1;
	// g r g r g r
	// b g b g b g
	// g r g r g r
	// b g b g b g
	const size_t nShifts = 8;
	const size_t nShifts_p1 = nShifts + 1;
	const size_t nShifts_p2 = nShifts + 2;
	if (warpIndex & 1) {
		for (int y=0; y<y_loop_count; ++y) {
			const uint16_t* pUp = pSrc;
			const uint16_t* pMi = OFFSET_PTR(pSrc, src_pitch);
			const uint16_t* pLo = OFFSET_PTR(pSrc, 2 * src_pitch);
			uchar4* pDst0 = pDst;
			for (int x=0; x<x_loop_count; ++x) {
				uint16_t m1 = pMi[1];
				uint16_t m2 = pMi[2];
				pDst0->x = m1 >> nShifts;
				pDst0->y = (pUp[1] + pMi[0] + m2 + pLo[1] + 2) >> nShifts_p2;
				pDst0->z = (pUp[0] + pUp[2] + pLo[0] + pLo[2] + 2) >> nShifts_p2;
				++pDst0;

				pDst0->x = (m1 + pMi[3] + 1) >> nShifts_p1;
				pDst0->y = m2 >> nShifts;
				pDst0->z = (pUp[2] + pLo[2] + 1) >> nShifts_p1;
				++pDst0;

				pUp += x_step;
				pMi += x_step;
				pLo += x_step;
				pDst0 += x_step - 2;
			}
			pSrc = OFFSET_PTR(pSrc, y_stride * src_pitch);
			pDst = OFFSET_PTR(pDst, y_stride * dst_pitch);
		}
	}else {
		for (int y=0; y<y_loop_count; ++y) {
			const uint16_t* pUp = pSrc;
			const uint16_t* pMi = OFFSET_PTR(pSrc, src_pitch);
			const uint16_t* pLo = OFFSET_PTR(pSrc, 2 * src_pitch);
			uchar4* pDst0 = pDst;
			for (int x=0; x<x_loop_count; ++x) {
				uint32_t r0 = pUp[1] + pLo[1] + 1;
				uint16_t m0 = pMi[0];
				uint16_t m1 = pMi[1];
				uint16_t m2 = pMi[2];
				uint16_t m3 = pMi[3];
				pDst0->x = r0 >> nShifts_p1;
				pDst0->y = m1 >> nShifts;
				pDst0->z = (m0 + m2 + 1) >> nShifts_p1;
				++pDst0;

				pDst0->x = (r0 + pUp[3] + pLo[3] + 1) >> nShifts_p2;
				pDst0->y = (m1 + m3 + 1) >> nShifts_p1;
				pDst0->z = m2 >> nShifts;
				++pDst0;

				pUp += x_step;
				pMi += x_step;
				pLo += x_step;
				pDst0 += x_step - 2;
			}
			pSrc = OFFSET_PTR(pSrc, y_stride * src_pitch);
			pDst = OFFSET_PTR(pDst, y_stride * dst_pitch);
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
	uchar4* d_dst;
	size_t d_src_pitch, d_dst_pitch;
	ret = cudaHostRegister((void*)pSrc, width*height*2, cudaHostRegisterDefault);
	ret = cudaHostRegister((void*)pDst, width*height*4, cudaHostRegisterDefault);
	ret = cudaMallocPitch((void**)&d_src, &d_src_pitch, width*sizeof(uint16_t), height);
	ret = cudaMallocPitch((void**)&d_dst, &d_dst_pitch, width*sizeof(uchar4), height);
printf("cudaMalloc Elapsed %f\n", t.ElapsedSecond());
t.Start();
	ret = cudaMemcpy2D(d_src, d_src_pitch, pSrc, width*2, width*2, height, cudaMemcpyHostToDevice);
	cudaMemset(d_dst, 0, d_dst_pitch * height);
printf("cudaMemcpy Elapsed %f\n", t.ElapsedSecond());
t.Start();

	int numBlocksInAGrid = 64;
	int numThreadsInABlock = 512;

t.Start();
	demosaic<<<numBlocksInAGrid,numThreadsInABlock>>>(d_src, d_dst, width, height, d_src_pitch, d_dst_pitch);
	ret = cudaThreadSynchronize();
printf("demosaic Elapsed %f\n", t.ElapsedSecond());
t.Start();
	ret = cudaMemcpy2D(pDst, width*4, d_dst, d_dst_pitch, width*4, height, cudaMemcpyDeviceToHost);
printf("cudaMemcpy Elapsed %f\n", t.ElapsedSecond());

	ret = cudaFree(d_src);
	ret = cudaFree(d_dst);
}
