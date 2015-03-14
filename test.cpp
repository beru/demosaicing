#include "test.h"

#include <vector>
#include "demosaic.h"

struct Pixel
{
	uint16_t x;
	uint16_t y;
	uint16_t value;
};

void test(size_t width, size_t height, const uint16_t* image)
{
	std::vector<uint16_t> work(width * height);
	uint16_t* pWork = &work[0];

	for (size_t y=0; y<height; ++y) {
		for (size_t x=0; x<width; ++x) {
			size_t idx = y * width + x;
			pWork[idx] = image[idx] > 53220 ? -1 : 0;
		}
	}	
	
	{
		std::vector<uint32_t> work(width * height);
		uint32_t* pColor = &work[0];
		demosaic_grbg(image, width, height, pColor);
		int hoge = 0;
	}
	
}

