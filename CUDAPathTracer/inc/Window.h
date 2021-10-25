#pragma once
#include<Windows.h>

class DXSample;
class Window
{
public:
	static int Run(DXSample* sample, HINSTANCE handleInst, int nCmdShow);
	static HWND GetHwnd() { return mHandle; }
private:

	static LRESULT __stdcall WndProc(HWND hWnd, uint message, WPARAM wParam, LPARAM lParam);
	static HWND mHandle;
};