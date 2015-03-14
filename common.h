#pragma once

#include <stdio.h>
#include <tchar.h>
#include <assert.h>
#include "stdint.h"

template <typename T>
__forceinline void OffsetPtr(T*& ptr, ptrdiff_t offsetBytes)
{
	ptr = (T*) ((const char*)ptr + offsetBytes);
}

template <typename T, size_t N>
size_t countof( T (&array)[N] )
{
    return N;
}
