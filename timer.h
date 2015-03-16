#pragma once
#include <windows.h>

/*!
	@file   Timer.h
	@brief  ŽžŠÔŒv‘ª—p‚ÌClass
*/

class Timer
{
private:
	LARGE_INTEGER start_;

public:
	Timer()
	{
		Start();
	}

	void Start()
	{
		QueryPerformanceCounter(&start_);
	}

	static __int64 GetFrequency() {
		LARGE_INTEGER freq;
		QueryPerformanceFrequency(&freq);
		return freq.QuadPart;
	}
	
	__int64 Elapsed() const
	{
		LARGE_INTEGER now;
		QueryPerformanceCounter(&now);
		return now.QuadPart - start_.QuadPart;
	}
	
	double ElapsedSecond() const
	{
		return Elapsed() / (double) GetFrequency();
	}
	
};

