#include "test.h"

#include <vector>
#include "demosaic.h"
#include "Timer.h"

struct Pixel
{
	uint16_t x;
	uint16_t y;
	uint16_t value;
};

void test(size_t width, size_t height, const uint16_t* image)
{
	//std::vector<uint16_t> work0(width * (height + 2));
	//uint16_t* pWork = &work0[0];
	//float scales[] = {
	//	10.3566284,
	//	4.44606495,
	//	5.27747297,
	//	4.44606495,
	//};
	//for (size_t i=0; i<height; ++i) {
	//	for (size_t j=0; j<width; ++j) {
	//		int tmp = image[i*width+j] - 1023;
	//		int scale_idx = ((i & 1) * 2) + (j & 1); 
	//		tmp *= scales[scale_idx];
	//		pWork[i*width+j] = tmp;
	//	}
	//}
	
	{
		std::vector<uint32_t> work(width * height);
		uint32_t* pColor = &work[0];

#if 0
		Timer t;
		demosaic_grbg(image+width, width, height, pColor);
		printf("Elapsed %f\n", t.ElapsedSecond());
#else
		cuda_demosaic_grbg(image+width, width, height-1, pColor);
#endif
		printf("%p\n", pColor);
		int hoge = 0;
	}
	
}

