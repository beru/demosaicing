
#include "File.h"

#include <io.h>
#include <assert.h>
#include <memory.h>
#include <stdio.h>

#include <windows.h>

File::File(HANDLE hFile)
	:
	hFile_(hFile)
{
}

File::File(FILE* pFile)
{
	hFile_ = (HANDLE) _get_osfhandle(_fileno(pFile));
}

bool File::Read(void* pBuffer, size_t nNumberOfBytesToRead, size_t& nNumberOfBytesRead)
{
	bool ret = ReadFile(hFile_, pBuffer, nNumberOfBytesToRead, (LPDWORD)&nNumberOfBytesRead, NULL) != FALSE;
	if (!ret) {
		DWORD err = GetLastError();
		int err2 = 0;
		//FormatMessage(
	}
	return ret;
}

bool File::Write(const void* pBuffer, size_t nNumberOfBytesToWrite, size_t& nNumberOfBytesWritten)
{
	return WriteFile(hFile_, pBuffer, nNumberOfBytesToWrite, (LPDWORD)&nNumberOfBytesWritten, NULL) != FALSE;
}

size_t File::Seek(long lDistanceToMove, size_t dwMoveMethod)
{
	return SetFilePointer(hFile_, lDistanceToMove, NULL, dwMoveMethod);
}

size_t File::Tell() const
{
	/*
		-- Reference --
		http://nukz.net/reference/fileio/hh/winbase/filesio_3vhu.htm
	*/
	return SetFilePointer(
		hFile_, // must have GENERIC_READ and/or GENERIC_WRITE 
		0,     // do not move pointer 
		NULL,  // hFile is not large enough to need this pointer 
		FILE_CURRENT
	);  // provides offset from current position 
}

size_t File::Size() const
{
	return GetFileSize(hFile_, NULL);
}

bool File::Flush()
{
	return FlushFileBuffers(hFile_) != FALSE;
}

