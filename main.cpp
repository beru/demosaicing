
#include <stdio.h>
#include <stdint.h>
#include <array>
#include <vector>
#include "ReadImage\ReadImage.h"
#include "ReadImage\File.h"
#include "test.h"

#define WIDTH 5796
#define HEIGHT 3734

uint16_t values[WIDTH*HEIGHT];

int main(int argc, char* argv[])
{
	if (argc < 2) {
		return 0;
	}
	FILE* fp = fopen(argv[1], "rb");

#if 0
	File file(fp);
	ImageInfo info;
	if (ReadImageInfo(file, info)) {
		std::vector<uint8_t> image(info.width * info.height);
		ReadImageData(file, &image[0], info.width, 0);
		test(info.width, info.height, &image[0]);
	}
#else
	fread(values, 2, WIDTH*HEIGHT, fp);
	test(WIDTH, HEIGHT, values);

#endif
	return 0;
}

