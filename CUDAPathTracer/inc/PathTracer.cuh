#include"inc/DXSample.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include"cuda.h"
#include"cudaD3D11.h"

#include<iostream>


using namespace DirectX;

struct Ray
{
	__host__ __device__ Ray(XMFLOAT3 origin, XMFLOAT3 direction)
		: Origin(origin), Direction(direction)
	{}

	XMFLOAT3 Origin;
	XMFLOAT3 Direction;
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
	XMFLOAT3 mMeshBoundingBox[2];

	void startCuda();
	void loadAssets();

	void extractTrianglesFromVertices(std::vector<Vertex>& vertices, std::vector<uint>& indices, std::vector<XMFLOAT4> triangles);

	cudaExternalMemory_t importNTHandle(HANDLE handle, address64 size);
};