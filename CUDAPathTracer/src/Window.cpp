#include"inc/stdafx.h"
#include"inc/Window.h"
#include"inc/DXSample.h"


HWND Window::mHandle = nullptr;

int Window::Run(DXSample* sample, HINSTANCE handleInst, int nCmdShow)
{

	// sample init ...

	WNDCLASSEX wndClass = {};

	wndClass.cbSize = sizeof(WNDCLASSEX);
	wndClass.lpfnWndProc = WndProc;
	wndClass.hCursor = LoadCursor(nullptr, IDC_ARROW);
	wndClass.hInstance = handleInst;
	wndClass.lpszClassName = "DXSampleClass";
	wndClass.style = CS_HREDRAW | CS_VREDRAW;

	RegisterClassExA(&wndClass);

	RECT windowRect = { 0, 0, sample->GetWidth(), sample->GetHeight() };

	AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, false);

	mHandle = CreateWindow(wndClass.lpszClassName, sample->GetName(),
		WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
		windowRect.right - windowRect.left, windowRect.bottom - windowRect.top,
		nullptr, nullptr, handleInst, sample);


	sample->Awake();

	ShowWindow(mHandle, nCmdShow);

	MSG msg{};

	while (msg.message != WM_QUIT)
	{
		if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

	sample->Release();


	// sample destroy ...

	return static_cast<char>(msg.wParam);
}

LRESULT __stdcall Window::WndProc(HWND hWnd, uint message, WPARAM wParam, LPARAM lParam)
{
	DXSample* sample = reinterpret_cast<DXSample*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));

	switch (message)
	{
	case WM_CREATE:
	{
		LPCREATESTRUCT createStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
		SetWindowLongPtrA(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(createStruct->lpCreateParams));
	}
	return 0;

	case WM_KEYDOWN:
	{
		if (sample != nullptr)
		{
			// key pressed ...
		}
	}
	return 0;

	case WM_KEYUP:
	{
		if (sample != nullptr)
		{
			// key released ...
		}
	}
	return 0;

	case WM_PAINT:
	{
		if (sample != nullptr)
		{
			sample->Update(0.0f);
			sample->Render(0.0f);
			// Engine do ...
		}
	}
	return 0;

	case WM_DESTROY:
	{
		PostQuitMessage(0);
	}
	return 0;

	}

	return DefWindowProc(hWnd, message, wParam, lParam);
}
