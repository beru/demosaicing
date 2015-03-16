
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
	if ((d_part_length * (1 + threadIdx.x) - 1) >= width) {
		return;
	}

	pSrc += threadIdx.x * d_part_length;
	pDst += threadIdx.x * d_part_length;
	pSrc += width * blockIdx.x;
	pDst += width * (blockIdx.x + 1) + 1;

	// g r g r g r
	// b g b g b g
	// g r g r g r
	// b g b g b g
	const int nShifts = 8;
	uint16_t
		ul, uc, ur, urr,
		ml, mc, mr, mrr,
		ll, lc, lr, lrr
	;
	uint32_t r, g, b;
	if (blockIdx.x == 0) {
		for (size_t y=0; y<height; y+=2) {
			const uint16_t* pUp = pSrc;
			const uint16_t* pMi = pSrc + width;
			const uint16_t* pLo = pSrc + 2 * width;
			uchar4* pDst0 = pDst;
			ul = pUp[0]; uc = pUp[1]; ur = pUp[2];
			ml = pMi[0]; mc = pMi[1]; mr = pMi[2];
			ll = pLo[0]; lc = pLo[1]; lr = pLo[2];
			for (size_t x=0; x<d_part_length; x+=2) {
				uint32_t r0 = uc + lc + 1;
				r = r0 >> (1 + nShifts);
				g = mc >> nShifts;
				b = (ml + mr + 1) >> (1 + nShifts);
				pDst0->w = 0;
				pDst0->x = r;
				pDst0->y = g;
				pDst0->z = b;
				++pDst0;

				urr = pUp[3];
				mrr = pMi[3];
				lrr = pLo[3];

				r = (r0 + urr + lrr + 1) >> (2 + nShifts);
				g = (mc + mrr + 1) >> (1 + nShifts);
				b = mr >> nShifts;
				pDst0->w = 0;
				pDst0->x = r;
				pDst0->y = g;
				pDst0->z = b;
				++pDst0;

				ul = ur;
				ml = mr;
				ll = lr;
				uc = urr;
				mc = mrr;
				lc = lrr;
				ur = pUp[4];
				mr = pMi[4];
				lr = pLo[4];

				pUp += 2;
				pMi += 2;
				pLo += 2;
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
			ul = pUp[0]; uc = pUp[1]; ur = pUp[2];
			ml = pMi[0]; mc = pMi[1]; mr = pMi[2];
			ll = pLo[0]; lc = pLo[1]; lr = pLo[2];
			for (size_t x=0; x<d_part_length; x+=2) {
				r = mc >> nShifts;
				g = (uc + ml + mr + lc + 2) >> (2 + nShifts);
				b = (ul + ur + ll + lr + 2) >> (2 + nShifts);
				pDst0->w = 0;
				pDst0->x = r;
				pDst0->y = g;
				pDst0->z = b;
				++pDst0;

				urr = pUp[3];
				mrr = pMi[3];
				lrr = pLo[3];

				r = (mc + mrr + 1) >> (1 + nShifts);
				g = mr >> nShifts;
				b = (ur + lr + 1) >> (1 + nShifts);
				pDst0->w = 0;
				pDst0->x = r;
				pDst0->y = g;
				pDst0->z = b;
				++pDst0;

				ul = ur;
				ml = mr;
				ll = lr;
				uc = urr;
				mc = mrr;
				lc = lrr;
				ur = pUp[4];
				mr = pMi[4];
				lr = pLo[4];

				pUp += 2;
				pMi += 2;
				pLo += 2;
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
//	ret = cudaMemcpy(d_dst, pDst, width*height*sizeof(uint32_t), cudaMemcpyHostToDevice);
printf("Elapsed %f\n", t.ElapsedSecond());
t.Start();

	int numBlocksInAGrid = 2;
	int numThreadsInABlock = 256;

	int len = (width / numThreadsInABlock + 1) & ~1;

	ret = cudaMemcpyToSymbol((const void*)&d_part_length, &len, sizeof(len));

t.Start();
	demosaic<<<numBlocksInAGrid,numThreadsInABlock>>>(d_src, width, height, d_dst);
printf("Elapsed %f\n", t.ElapsedSecond());
t.Start();
	ret = cudaMemcpy(h_dst, d_dst, width*height*sizeof(uchar4), cudaMemcpyDeviceToHost);
	//ret = cudaMemcpy(pDst, d_dst, width*height*sizeof(uchar4), cudaMemcpyDeviceToHost);
printf("Elapsed %f\n", t.ElapsedSecond());

	ret = cudaFree(d_src);
	ret = cudaFree(d_dst);
	ret = cudaFreeHost(h_dst);
}
