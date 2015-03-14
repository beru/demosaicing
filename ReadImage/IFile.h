#pragma once

/*!

概要
	画像の入出力用に仮想的にFileを扱えるようにする
	
制限
	非同期操作には対応しない
	2GBまで
	
備考
	CxImage by Davide Pizzolato (http://www.xdp.it/cximage.htm)	を参考にしました。

	InterfaceをWindowsAPIのFile関数に似せています

*/

class IFile
{
public:
	virtual bool Read(void* pBuffer, size_t nNumberOfBytesToRead, size_t& nNumberOfBytesRead) = 0;
	virtual bool Write(const void* pBuffer, size_t nNumberOfBytesToWrite, size_t& nNumberOfBytesWritten) = 0;
	virtual size_t Seek(long lDistanceToMove, size_t dwMoveMethod) = 0;
	virtual size_t Tell() const = 0;
	virtual size_t Size() const = 0;
	virtual bool IsEof() const { return Tell() == Size(); }
	virtual bool Flush() = 0;

	virtual bool HasBuffer() const = 0;
	virtual const void* GetBuffer() const = 0;
};

