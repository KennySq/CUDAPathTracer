#pragma once
#include<Windows.h>
#include<stdexcept>

typedef unsigned int uint;
typedef unsigned long long address64;

static void Throw(HRESULT result)
{
	if (FAILED(result))
	{
		throw std::runtime_error("");
	}
}

static std::wstring GetWorkingDirectoryW()
{
	static constexpr unsigned int MAX_STR = 512;
	wchar_t buffer[MAX_STR];

	GetModuleFileNameW(nullptr, buffer, MAX_STR);

	std::wstring path(buffer);

	path = path.substr(0, path.find_last_of('\\') + 1);

	return path;
}

static std::string GetWorkingDirectoryA()
{
	static constexpr unsigned int MAX_STR = 512;
	char buffer[MAX_STR];

	GetModuleFileName(nullptr, buffer, MAX_STR);

	std::string path(buffer);

	path = path.substr(0, path.find_last_of('\\') + 1);

	return path;
}