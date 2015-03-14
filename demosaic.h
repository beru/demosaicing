#pragma once

#include <stdint.h>

void demosaic_grbg(
	const uint16_t* pSrc,
	size_t width,
	size_t height,
	uint32_t* pDst
);

