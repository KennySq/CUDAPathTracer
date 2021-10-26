#include"inc/DXSample.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include"cudaTypedefs.h"
#include"cuda.h"
#include"cudaD3D11.h"
#include"cuda_surface_types.h"
#include"cuda_pipeline.h"


#include"Math.cuh"

#include<iostream>


using namespace DirectX;

struct Ray
{
	__host__ __device__ Ray(float3 origin, float3 direction)
		: Origin(origin), Direction(direction)
	{}

	float3 Origin;
	float3 Direction;
};

struct Triangle
{
	unsigned int i[3];
};

enum eGeometryType
{
	GEOMETRY_SPHERE,
	GEOMETRY_MESH,
};


class PathTracer : public DXSample
{
	struct ScreenVertex
	{
		XMFLOAT3 Position;
		XMFLOAT2 Texcoord;
	};

public:
	PathTracer();

	virtual void Awake() override;
	virtual void Update(float delta) override;
	virtual void Render(float delta) override;
	virtual void Release() override;
private:
	ComPtr<ID3D11Texture2D> mBackBuffer;
	ComPtr<ID3D11RenderTargetView> mBackBufferRTV;

	ComPtr<ID3D11Texture2D> mCudaSharedTexture;

	std::vector<XMFLOAT4> mMeshTriangles;
	float3 mMeshBoundingBox[2];

	ComPtr<ID3D11Texture2D> mRenderTexture;
	ComPtr<ID3D11ShaderResourceView> mRenderTextureSRV;
	ComPtr<ID3D11UnorderedAccessView> mRenderTextureUAV;
	CUgraphicsResource mCudaRenderTexture;

	CUsurfObject mCudaRenderSurface;
	
	CUdevice mCudaDevice;
	

	uint mTriangleCount;
	
	ComPtr<ID3D11Buffer> mScreenVB;
	ComPtr<ID3D11Buffer> mScreenIB;
	ComPtr<ID3D11VertexShader> mScreenVS;
	ComPtr<ID3D11PixelShader> mScreenPS;
	ComPtr<ID3D11InputLayout> mScreenIL;

	D3D11_VIEWPORT mViewport;

	uint hashFrame(uint frame);
	void loadAssets();
	void makeScreen();

	void drawScreen();

	void extractTrianglesFromVertices(std::vector<Vertex>& vertices, std::vector<uint>& indices, std::vector<XMFLOAT4> triangles);

	cudaExternalMemory_t importNTHandle(HANDLE handle, address64 size);

	uint mFrameIndex = 0;
};