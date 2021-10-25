#include"inc/stdafx.h"
#include"inc/PathTracer.cuh"
#include"inc/Window.h"
#include <stdio.h>


int WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int nCmdShow)
{
	PathTracer sample;

	return Window::Run(&sample, hInst, nCmdShow);
}