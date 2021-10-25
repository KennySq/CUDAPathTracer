#include"inc/PathTracer.cuh"

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

// General Variables for CUDA //

float* cudaTriangles = nullptr;

////////////////////////////////

PathTracer::PathTracer()
	: DXSample(1280, 720, "CUDA PathTracer")
{

}

void PathTracer::Awake()
{
	AcquireHardware();
	AllocConsole();

	loadAssets();
	startCuda();



	//importNTHandle();

}

void PathTracer::Update(float delta)
{
	mContext->ClearRenderTargetView(mBackBufferRTV.Get(), DirectX::Colors::Blue);

}

void PathTracer::Render(float delta)
{

	mSwapchain->Present(1, 0);
}

void PathTracer::Release()
{
}

__global__ void someGlobal()
{
	if (threadIdx.x == 0)
	{
		printf("%s\n", "Hello");
	}

	return;
}

void PathTracer::startCuda()
{
	someGlobal << < 1, 1, 1 >> > ();

}

void PathTracer::loadAssets()
{
	Throw(mSwapchain->GetBuffer(0, IID_PPV_ARGS(&mBackBuffer)));

	D3D11_RENDER_TARGET_VIEW_DESC rtvDesc{};

	rtvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;

	Throw(mDevice->CreateRenderTargetView(mBackBuffer.Get(), &rtvDesc, mBackBufferRTV.GetAddressOf()));

	D3D11_TEXTURE2D_DESC cudaTextureDesc{};

	cudaTextureDesc.Width = mWidth;
	cudaTextureDesc.Height = mHeight;
	cudaTextureDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	cudaTextureDesc.ArraySize = 1;
	cudaTextureDesc.MipLevels = 1;
	cudaTextureDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
	cudaTextureDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
	cudaTextureDesc.SampleDesc.Count = 1;

	Throw(mDevice->CreateTexture2D(&cudaTextureDesc, nullptr, mCudaSharedTexture.GetAddressOf()));

	std::string meshPath = GetWorkingDirectoryA();
	meshPath += "..\\..\\CUDAPathTracer\\resources\\shiba\\shiba.fbx";
	FbxLoader loader(meshPath.c_str());

	//extractTrianglesFromVertices(loader.Vertices, mMeshTriangles);


}

cudaExternalMemory_t PathTracer::importNTHandle(HANDLE handle, address64 size)
{
	cudaExternalMemory_t extMemory = nullptr;
	cudaExternalMemoryHandleDesc desc{};

	desc.size = size;
	desc.type = cudaExternalMemoryHandleTypeD3D11Resource;
	desc.handle.win32.handle = handle;
	desc.flags |= cudaExternalMemoryDedicated;

	cudaError_t error = cudaImportExternalMemory(&extMemory, &desc);
	std::cout << cudaGetErrorString(error) << '\n';

	CloseHandle(handle);

	return extMemory;
}

void PathTracer::extractTrianglesFromVertices(std::vector<Vertex>& vertices, std::vector<uint>& indices, std::vector<XMFLOAT4> triangles)
{
	for (uint i = 0; i < indices.size(); i += 3)
	{
		XMFLOAT4 position1 = XMFLOAT4(vertices[i].mPosition.x, vertices[i].mPosition.y, vertices[i].mPosition.z, 1.0f);
		XMFLOAT4 position2 = XMFLOAT4(vertices[i + 1].mPosition.x, vertices[i + 1].mPosition.y, vertices[i + 1].mPosition.z, 1.0f);
		XMFLOAT4 position3 = XMFLOAT4(vertices[i + 2].mPosition.x, vertices[i + 2].mPosition.y, vertices[i + 2].mPosition.z, 1.0f);

		// pre-edge calculation, edge를 미리 계산하여 Path-tracing 시의 GPU 연산 부하를 줄입니다.
		mMeshTriangles.push_back(XMFLOAT4(position1.x, position1.y, position1.z, 0.0f));
		mMeshTriangles.push_back(XMFLOAT4(position2.x - position1.x, position2.y - position1.y, position2.z - position1.z, 1.0f));
		mMeshTriangles.push_back(XMFLOAT4(position3.x - position2.x, position3.y - position2.y, position3.z - position2.z, 0.0f));
	}

	mMeshBoundingBox[0] = XMFLOAT3(1250.0f, 1250.0f, 1250.0f);
	mMeshBoundingBox[1] = XMFLOAT3(-1250.0f, -1250.0f, -1250.0f);

	long long triangleSize = indices.size() * sizeof(XMFLOAT3);
	uint triangleCount = indices.size() / 3;

	if (triangleSize > 0)
	{
		cudaMalloc(&cudaTriangles, triangleSize);

		cudaMemcpy(cudaTriangles, mMeshTriangles.data(), triangleSize, cudaMemcpyHostToDevice);

	}




}