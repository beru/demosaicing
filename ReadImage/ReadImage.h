#pragma once

#include "IFile.h"

struct ImageInfo
{
	size_t width;
	size_t height;
	size_t bitsPerSample;
	size_t samplesPerPixel;
};

bool ReadImageInfo(IFile& file, ImageInfo& info);
bool ReadImageData(IFile& file, unsigned char* dest, int lineOffset, void* palettes);

