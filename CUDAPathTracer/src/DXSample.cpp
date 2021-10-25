#include"inc/DXSample.h"
#include"inc/Window.h"
#include"inc/WinUtil.h"

DXSample::DXSample(uint width, uint height, const char* appName)
	: mWidth(width), mHeight(height), mAppName(appName)
{

}

void DXSample::AcquireHardware()
{
	CreateDXGIFactory(IID_PPV_ARGS(&mFactory));

	D3D_FEATURE_LEVEL selectedLevel;
	D3D_FEATURE_LEVEL featureLevels[] =
	{
		D3D_FEATURE_LEVEL_11_0,
		D3D_FEATURE_LEVEL_10_1,
		D3D_FEATURE_LEVEL_10_0
	};

#ifdef _DEBUG
	DWORD deviceFlag = D3D11_CREATE_DEVICE_DEBUG;
#endif

	Throw(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, deviceFlag, featureLevels, ARRAYSIZE(featureLevels), D3D11_SDK_VERSION, mDevice.GetAddressOf(), &selectedLevel, mContext.GetAddressOf()));

	DXGI_SWAP_CHAIN_DESC scDesc{};

	scDesc.BufferCount = 1;
	scDesc.BufferDesc.Width = mWidth;
	scDesc.BufferDesc.Height = mHeight;
	scDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	scDesc.SampleDesc.Count = 1;
	scDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
	scDesc.OutputWindow = Window::GetHwnd();
	scDesc.Windowed = true;
	scDesc.BufferDesc.RefreshRate.Denominator = 144;
	scDesc.BufferDesc.RefreshRate.Numerator = 1;
	scDesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
	scDesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
	
	Throw(mFactory->CreateSwapChain(mDevice.Get(), &scDesc, mSwapchain.GetAddressOf()));

	
}
