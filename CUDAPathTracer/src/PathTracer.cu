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
texture<float4, 1, cudaReadModeElementType> cudaTriangleTexture;

float3 cudaSceneBoundBoxMin;
float3 cudaSceneBoundBoxMax;

float4* cudaRenderTexture;

////////////////////////////////

void GetError(cudaError_t error)
{
	printf("%s\n", cudaGetErrorString(error));
}

PathTracer::PathTracer()
	: DXSample(1280, 720, "CUDA PathTracer")
{
	cudaDeviceSynchronize();
	//cuCtxCreate()
}

__global__ void kernelRender(CUsurfObject surf, float4* outTexture, const uint triCount, uint frameIndex, uint hashFrameIndex, float fovRadians, uint width, uint height, float3 sceneMinBound, float3 sceneMaxBound)
{
	uint x = blockDim.x * blockIdx.x + threadIdx.x; ///blockDim.x * blockIdx.y + threadIdx.x;
	uint y = blockDim.y * blockIdx.y + threadIdx.y;

	float3 camPos = make_float3(-25.0f, 0.0f, -25.0f);
	float3 camDir = Normalize(camPos - make_float3(0, 0, 0));

	Ray ray = Ray(camPos, camDir);
	
	float3 u = make_float3(width * fovRadians / height, 0.0f, 0.0f);
	float3 v = Normalize(Cross(u, camDir)) * fovRadians;

	float4 result = { 0,0,0,0 };

	int pixelIndex = (height - y - 1) * width + x;

	float4 value = make_float4(0, 1, 1, 0);
	surf2Dwrite<float4>(value, surf, x * sizeof(float4), y);
}	

void PathTracer::Awake()
{
	AcquireHardware();
	AllocConsole();

	loadAssets();


	//importNTHandle();

}

void PathTracer::Update(float delta)
{
	cudaThreadSynchronize();

	dim3 block = dim3(16, 16, 1);
	dim3 grid = dim3(mWidth / block.x, mHeight / block.y, 1);

	uint hashedFrame = hashFrame(mFrameIndex);

	kernelRender << < grid, block >> > (mCudaRenderSurface, cudaRenderTexture, mTriangleCount, mFrameIndex, hashedFrame, XMConvertToRadians(90.0f), mWidth, mHeight, cudaSceneBoundBoxMin, cudaSceneBoundBoxMax);

	cudaThreadSynchronize();

	mContext->ClearRenderTargetView(mBackBufferRTV.Get(), DirectX::Colors::Blue);
}

void PathTracer::Render(float delta)
{
	cudaError_t error = cudaGetLastError();
	std::cout <<cudaGetErrorString(error) << '\n';


	drawScreen();
	mSwapchain->Present(1, 0);
	mFrameIndex++;

}

void PathTracer::Release()
{
	cuGraphicsUnmapResources(1, &mCudaRenderTexture, 0);
	cuSurfObjectDestroy(mCudaRenderSurface);
}

__global__ void someGlobal()
{
	if (threadIdx.x == 0)
	{
		printf("%s\n", "Hello");
	}

	return;
}

uint PathTracer::hashFrame(uint frame)
{
	frame = (frame ^ 61) ^ (frame >> 16);
	frame = frame + (frame << 3);
	frame = frame ^ (frame >> 4);
	frame = frame * 0x27d4eb2d;
	frame = frame ^ (frame >> 15);

	return frame;
}

void PathTracer::loadAssets()
{
	mViewport = {};
	mViewport.Width = mWidth;
	mViewport.Height = mHeight;
	mViewport.MaxDepth = 1.0f;

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
	cudaTextureDesc.SampleDesc.Count = 1;

	Throw(mDevice->CreateTexture2D(&cudaTextureDesc, nullptr, mCudaSharedTexture.GetAddressOf()));
	
	D3D11_TEXTURE2D_DESC cudaRenderTextureDesc{};

	cudaRenderTextureDesc.Width = mWidth;
	cudaRenderTextureDesc.Height = mHeight;
	cudaRenderTextureDesc.MipLevels = 1;
	cudaRenderTextureDesc.ArraySize = 1;
	cudaRenderTextureDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
	cudaRenderTextureDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	cudaRenderTextureDesc.SampleDesc.Count = 1;
	
	Throw(mDevice->CreateTexture2D(&cudaRenderTextureDesc, nullptr, mRenderTexture.GetAddressOf()));
	

	D3D11_SHADER_RESOURCE_VIEW_DESC renderTextureSrvDesc{};

	renderTextureSrvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	renderTextureSrvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	renderTextureSrvDesc.Texture2D.MipLevels = 1;
	Throw(mDevice->CreateShaderResourceView(mRenderTexture.Get(), &renderTextureSrvDesc, mRenderTextureSRV.GetAddressOf()));

	D3D11_UNORDERED_ACCESS_VIEW_DESC renderTextureUavDesc{};

	renderTextureUavDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	renderTextureUavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
	renderTextureUavDesc.Texture2D.MipSlice = 0;

	Throw(mDevice->CreateUnorderedAccessView(mRenderTexture.Get(), &renderTextureUavDesc, mRenderTextureUAV.GetAddressOf()));
	
	
	CUresult curesult;
	curesult = cuGraphicsD3D11RegisterResource(&mCudaRenderTexture, mRenderTexture.Get(), CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
	cuGraphicsResourceSetMapFlags(mCudaRenderTexture, CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD);


	const char* errorStr = "Hello";
	curesult = cuGraphicsMapResources(1, &mCudaRenderTexture, 0);
	cuGetErrorString(curesult, &errorStr);

	CUarray retArray;
	curesult = cuGraphicsSubResourceGetMappedArray(&retArray, mCudaRenderTexture, 0, 0);
	

	cuGetErrorString(curesult, &errorStr);

	
	CUDA_RESOURCE_DESC cuReourceDesc{};
	
	cuReourceDesc.resType = CU_RESOURCE_TYPE_ARRAY;
	cuReourceDesc.res.array.hArray = retArray;
	cuSurfObjectCreate(&mCudaRenderSurface, &cuReourceDesc);
		



	std::string meshPath = GetWorkingDirectoryA();
	meshPath += "..\\..\\CUDAPathTracer\\resources\\shiba\\shiba.fbx";
	FbxLoader loader(meshPath.c_str());

	extractTrianglesFromVertices(loader.Vertices, loader.Indices, mMeshTriangles);

	makeScreen();

}

void PathTracer::makeScreen()
{
	ScreenVertex vertices[] =
	{
		{ {-1, -1, 0}, {0,0}},
		{ {1, -1, 0}, {1,0}},
		{ {-1, 1, 0}, {0,1}},
		{ {1, 1, 0}, {1,1}},
	};

	uint indices[] =
	{
		2,1,0,
		2,3,1
	};
	
	D3D11_BUFFER_DESC vbDesc{}, ibDesc{};
	D3D11_SUBRESOURCE_DATA vbSub{}, ibSub{};

	vbDesc.ByteWidth = sizeof(ScreenVertex) * 4;
	vbDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbSub.pSysMem = vertices;

	ibDesc.ByteWidth = sizeof(unsigned int) * 6;
	ibDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	ibSub.pSysMem = indices;

	D3D11_INPUT_ELEMENT_DESC inputElements[] =
	{
		{"POSITION",0, DXGI_FORMAT_R32G32B32_FLOAT, 0,0, D3D11_INPUT_PER_VERTEX_DATA, 0},
		{"TEXCOORD",0, DXGI_FORMAT_R32G32_FLOAT, 0,0xFFFFFFFF, D3D11_INPUT_PER_VERTEX_DATA, 0},
	};

	ID3DBlob* vBlob, * pBlob, * errBlob;

	Throw(mDevice->CreateBuffer(&vbDesc, &vbSub, mScreenVB.GetAddressOf()));
	Throw(mDevice->CreateBuffer(&ibDesc, &ibSub, mScreenIB.GetAddressOf()));

	std::wstring shaderPath = GetWorkingDirectoryW();

	shaderPath += L"..\\..\\CUDAPathTracer\\resources\\Screen.hlsl";

#ifdef _DEBUG
	DWORD compileFlag = D3DCOMPILE_DEBUG;
#else
	DWORD compileFlag = 0;

#endif

	Throw(D3DCompileFromFile(shaderPath.c_str(), nullptr, nullptr, "vert", "vs_4_0", compileFlag, 0, &vBlob, &errBlob));
	Throw(D3DCompileFromFile(shaderPath.c_str(), nullptr, nullptr, "frag", "ps_5_0", compileFlag, 0, &pBlob, &errBlob));
	Throw(mDevice->CreateVertexShader(vBlob->GetBufferPointer(), vBlob->GetBufferSize(), nullptr, mScreenVS.GetAddressOf()));
	Throw(mDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), nullptr, mScreenPS.GetAddressOf()));
	Throw(mDevice->CreateInputLayout(inputElements, 2, vBlob->GetBufferPointer(), vBlob->GetBufferSize(), mScreenIL.GetAddressOf()));
}

void PathTracer::drawScreen()
{
	static ID3D11ShaderResourceView* nullSrv[] = { nullptr };
	static ID3D11RenderTargetView* nullRtv[] = { nullptr };
	static uint strides[] = {sizeof(ScreenVertex) };
	static uint offsets[] = { 0 };

	mContext->IASetVertexBuffers(0, 1, mScreenVB.GetAddressOf(), strides, offsets);

	mContext->IASetIndexBuffer(mScreenIB.Get(), DXGI_FORMAT_R32_UINT, 0);
	mContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	mContext->IASetInputLayout(mScreenIL.Get());

	mContext->OMSetRenderTargets(1, mBackBufferRTV.GetAddressOf(), nullptr);

	mContext->VSSetShader(mScreenVS.Get(), nullptr, 0);
	mContext->PSSetShader(mScreenPS.Get(), nullptr, 0);
	mContext->PSSetShaderResources(0, 1, mRenderTextureSRV.GetAddressOf());

	mContext->DrawIndexed(6, 0, 0);

	mContext->OMSetRenderTargets(1, nullRtv, nullptr);

	mContext->RSSetViewports(1, &mViewport);

	mContext->PSSetShaderResources(0, 1, nullSrv);
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

extern "C"
{
	void InitTriangleTexture(float* triangles, uint triangleCount)
	{
		cudaTriangleTexture.normalized = false;
		cudaTriangleTexture.filterMode = cudaFilterModePoint;
		cudaTriangleTexture.addressMode[0] = cudaAddressModeWrap;

		long long size = sizeof(XMFLOAT4) * triangleCount * 3;

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<XMFLOAT4>();

		cudaError_t error = cudaBindTexture(nullptr, &cudaTriangleTexture, triangles, &channelDesc, size);
		GetError(error);

	}
}

void PathTracer::extractTrianglesFromVertices(std::vector<Vertex>& vertices, std::vector<uint>& indices, std::vector<XMFLOAT4> triangles)
{
	for (uint i = 0; i < indices.size(); i += 3)
	{
		uint index1 = indices[i];
		uint index2 = indices[i+1];
		uint index3 = indices[i+2];

		XMFLOAT4 position1 = XMFLOAT4(vertices[index1].mPosition.x, vertices[index1].mPosition.y, vertices[index1].mPosition.z, 1.0f);
		XMFLOAT4 position2 = XMFLOAT4(vertices[index2].mPosition.x, vertices[index2].mPosition.y, vertices[index2].mPosition.z, 1.0f);
		XMFLOAT4 position3 = XMFLOAT4(vertices[index3].mPosition.x, vertices[index3].mPosition.y, vertices[index3].mPosition.z, 1.0f);

		// pre-edge calculation, edge를 미리 계산하여 Path-tracing 시의 GPU 연산 부하를 줄입니다.
		mMeshTriangles.push_back(XMFLOAT4(position1.x, position1.y, position1.z, 0.0f));
		mMeshTriangles.push_back(XMFLOAT4(position2.x - position1.x, position2.y - position1.y, position2.z - position1.z, 1.0f));
		mMeshTriangles.push_back(XMFLOAT4(position3.x - position2.x, position3.y - position2.y, position3.z - position2.z, 0.0f));
	}

	mMeshBoundingBox[0] = make_float3(1250.0f, 1250.0f, 1250.0f);
	mMeshBoundingBox[1] = make_float3(-1250.0f, -1250.0f, -1250.0f);

	long long triangleSize = indices.size() * sizeof(XMFLOAT3);
	uint triangleCount = indices.size() / 3;

	if (triangleSize > 0)
	{
		cudaError_t error = cudaMalloc(&cudaTriangles, triangleSize);
		GetError(error);


		error = cudaMemcpy(cudaTriangles, mMeshTriangles.data(), triangleSize, cudaMemcpyHostToDevice);
		GetError(error);

		InitTriangleTexture(cudaTriangles, triangleCount);

	}

	cudaSceneBoundBoxMax = mMeshBoundingBox[0];
	cudaSceneBoundBoxMin = mMeshBoundingBox[1];


}