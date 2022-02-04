
#include"DXSample.h"
#include"Math.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include"cudaTypedefs.h"
#include"cuda.h"
#include"cudaD3D11.h"
#include"cuda_surface_types.h"
#include"cuda_pipeline.h"
#include"curand_globals.h"
#include"curand_kernel.h"
#include"surface_functions.h"
#pragma comment(lib, "cudart.lib")

struct ScreenVertex
{
	XMFLOAT3 Position;
	XMFLOAT2 Texcoord;
};

struct Triangle
{
	unsigned int Index[3];
};

struct PathTracer : public DXSample
{
public:

	PathTracer()
		: DXSample(1280, 720, "CUDA Path Tracer")
	{
	}

	virtual void Awake() override;
	virtual void Update(float delta) override;
	virtual void Render(float delta) override;
	virtual void Release() override;

private:
	void getGeometryFromMesh();

	void drawScreen();
	void loadAssets();

	ComPtr<ID3D11Texture2D> mBackBuffer;
	ComPtr<ID3D11RenderTargetView> mBackBufferRTV;
	
	ComPtr<ID3D11Texture2D> mRenderTexture;
	ComPtr<ID3D11ShaderResourceView> mRenderTextureSRV;
	ComPtr<ID3D11UnorderedAccessView> mRenderTextureUAV;

	CUgraphicsResource mCudaRenderTexture;
	CUsurfObject mCudaRenderSurface;
	CUdevice mCudaDevice;

	ComPtr<ID3D11Texture2D> mCudaSharedTexture;

	ComPtr<ID3D11Buffer> mScreenVB;
	ComPtr<ID3D11Buffer> mScreenIB;

	ComPtr<ID3D11VertexShader> mScreenVS;
	ComPtr<ID3D11PixelShader> mScreenPS;
	ComPtr<ID3D11InputLayout> mScreenIL;

	D3D11_VIEWPORT mViewport;
	
	size_t mFrameIndex = 0;

};

