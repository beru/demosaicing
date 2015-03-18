#pragma once

#include <stdint.h>

void demosaic_grbg(
	const uint16_t* pSrc,
	size_t width,
	size_t height,
	uint32_t* pDst
);

void cuda_demosaic_grbg(
	const uint16_t* __restrict pSrc,
	size_t width,
	size_t height,
	uint32_t* __restrict pDst
);
