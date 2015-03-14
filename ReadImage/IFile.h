#pragma once

/*!

�T�v
	�摜�̓��o�͗p�ɉ��z�I��File��������悤�ɂ���
	
����
	�񓯊�����ɂ͑Ή����Ȃ�
	2GB�܂�
	
���l
	CxImage by Davide Pizzolato (http://www.xdp.it/cximage.htm)	���Q�l�ɂ��܂����B

	Interface��WindowsAPI��File�֐��Ɏ����Ă��܂�

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

