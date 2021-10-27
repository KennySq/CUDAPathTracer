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

enum Refl_t
{
	DIFF, // Diffuse
	SPEC, // Specular
	REFR, // Refract
};

struct Sphere {

	float rad;				// radius 
	float3 pos, emi, col;	// position, emission, color 
	Refl_t refl;			// reflection type (DIFFuse, SPECular, REFRactive)

	__device__ float intersect(const Ray& r) const { // returns distance, 0 if nohit 

		// Ray/sphere intersection
		// Quadratic formula required to solve ax^2 + bx + c = 0 
		// Solution x = (-b +- sqrt(b*b - 4ac)) / 2a
		// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 

		float3 op = pos - r.Origin;  // 
		float t, epsilon = 0.01f;
		float b = Dot(op, r.Direction);
		float disc = b * b - Dot(op, op) + rad * rad; // discriminant
		if (disc < 0) return 0; else disc = sqrtf(disc);
		return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);
	}
};

__constant__ Sphere spheres[] = {
	// FORMAT: { float radius, float3 position, float3 emission, float3 colour, Refl_t material }
	// cornell box
	//{ 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, DIFF }, //Left 1e5f
	//{ 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF }, //Right 
	//{ 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Back 
	//{ 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 0.00f, 0.00f, 0.00f }, DIFF }, //Front 
	//{ 1e5f, { 50.0f, -1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Bottom 
	//{ 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top 
	//{ 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 0.99f, 0.99f, 0.99f }, SPEC }, // small sphere 1
	//{ 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.f, .0f }, { 0.09f, 0.49f, 0.3f }, REFR }, // small sphere 2
	//{ 600.0f, { 50.0f, 681.6f - .5f, 81.6f }, { 3.0f, 2.5f, 2.0f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light 12, 10 ,8

	//outdoor scene: radius, position, emission, color, material

	//{ 1600, { 3000.0f, 10, 6000 }, { 37, 34, 30 }, { 0.f, 0.f, 0.f }, DIFF },  // 37, 34, 30 // sun
	//{ 1560, { 3500.0f, 0, 7000 }, { 50, 25, 2.5 }, { 0.f, 0.f, 0.f }, DIFF },  //  150, 75, 7.5 // sun 2
	{ 10000, { 50.0f, 40.8f, -1060 }, { 0.0003, 0.01, 0.15 }, { 0.175f, 0.175f, 0.25f }, DIFF }, // sky
	{ 100000, { 50.0f, -100000, 0 }, { 0.0, 0.0, 0 }, { 0.8f, 0.2f, 0.f }, DIFF }, // ground
	{ 110000, { 50.0f, -110048.5, 0 }, { 3.6, 2.0, 0.2 }, { 0.f, 0.f, 0.f }, DIFF },  // horizon brightener
	{ 4e4, { 50.0f, -4e4 - 30, -3000 }, { 0, 0, 0 }, { 0.2f, 0.2f, 0.2f }, DIFF }, // mountains
	{ 82.5, { 30.0f, 180.5, 42 }, { 16, 12, 6 }, { .6f, .6f, 0.6f }, DIFF },  // small sphere 1
	{ 12, { 115.0f, 10, 105 }, { 0.0, 0.0, 0.0 }, { 0.9f, 0.9f, 0.9f }, REFR },  // small sphere 2
	{ 22, { 65.0f, 22, 24 }, { 0, 0, 0 }, { 0.9f, 0.9f, 0.9f }, SPEC }, // small sphere 3
};

////////////////////////////////

void GetError(cudaError_t error)
{
	printf("%s\n", cudaGetErrorString(error));
}

PathTracer::PathTracer()
	: DXSample(1280, 720, "CUDA PathTracer")
{
	cudaDeviceSynchronize();
}

__device__ float3 radiance(Ray& r, curandState* randState, const int triangleCount, const float3& sceneMinBound, const float3& sceneMaxBound)
{
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);
	float3 tempColor = make_float3(0.0f, 0.0f, 0.0f);

	float t = 100000;
	int sphereID = -1;
	int triangleID = -1;

	float3 f; // primitive color?
	float3 emit; // primitive emissive color?
	float3 hit; //intersection point
	float3 normal; // normal
	float3 orientedNormal; // corrected normal
	float3 d; // next ray direction
	
	if (intersectScene(r, t, sphereID, triangleID, triangleCount, sceneMinBound, sceneMaxBound) == false)
	{
		return make_float3(0, 0, 0);
	}

	Sphere& sphere = spheres[sphereID];

	hit = r.Origin + r.Direction * t;
	normal = Normalize(hit - sphere.pos);

	orientedNormal = Dot(normal, r.Direction) < 0 ? normal : normal * -1;
	f = sphere.col;
	emit = sphere.emi;

	tempColor += (mask * emit);

	
}

__device__ inline bool intersectScene(const Ray& r, float& t, int& sphereID, int& triangleID, const int triCount, const float3& boundMin, const float3& boundMax)
{
	float tMin = 1e20;
	float tMax = -1e20;

	float d = 1e21;
	float k = 1e21;
	float q = 1e21;

	float inf = t = 1e20;

	float sphereCount = sizeof(spheres) / sizeof(Sphere);

	for (int i = int(sphereCount); i > 0; i--)
	{
		

		if ((d = spheres[i].intersect(r)) && k < t)
		{
			t = k;
			sphereID = i;
		}
	}



}

__global__ void kernelRender(CUsurfObject surf, float4* outTexture, const uint triCount, uint frameIndex, uint hashFrameIndex, float fovRadians, uint width, uint height, float3 sceneMinBound, float3 sceneMaxBound)
{
	uint x = blockDim.x * blockIdx.x + threadIdx.x; ///blockDim.x * blockIdx.y + threadIdx.x;
	uint y = blockDim.y * blockIdx.y + threadIdx.y;

	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockIdx.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	float3 camPos = make_float3(25.0f, 0.0f, 25.0f);
	float3 camDir = Normalize(camPos - make_float3(0, 0, 0));

	Ray ray = Ray(camPos, camDir);
	
	float3 u = make_float3(width * fovRadians / height, 0.0f, 0.0f);
	float3 v = Normalize(Cross(u, camDir)) * fovRadians;

	float4 result = { 0,0,0,0 };

	int pixelIndex = (height - y - 1) * width + x;

	float4 value = make_float4(0, 1, 1, 0);

	curandState randState;
	curand_init(hashFrameIndex + threadId, 0, 0, &randState);

	Ray r = { camPos, camDir };

	value = make_float4(v.x, v.y, 0.0f, 1.0f);

	for (uint i = 0; i < 1; i++)
	{
		float3 d = u * ((.25f + x) / width - .5) + v * ((.25 + y) / height - .5) + r.Direction;
		//float3 d = u * ((float)x / (float)width ) + v * ((float)y / (float)height ) + r.Direction;
	
		d = Normalize(d);


//		value = make_float4(d.x,d.y,d.z, 1.0f);
	
		float3 color = radiance(r, &randState, triCount, sceneMinBound, sceneMaxBound);
		value = make_float4(color.x, color.y, color.z, 1.0f);
		//value = make_float4(d.x, d.y, d.z, 1.0f);
	}



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

// 프레임 해싱에 대한 코드입니다.
// cuRand가 thread 마다 랜덤한 값을 부여해야하기 때문에
// frame 해싱 값을 기준으로 각 threadid를
// 더해 매 프레임마다, 각 스레드에 대해 고유한 값을 생성합니다.
// reference : https://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
uint PathTracer::hashFrame(uint frame)
{
	frame = (frame ^ 61) ^ (frame >> 16);
	frame = frame + (frame << 3);
	frame = frame ^ (frame >> 4);
	frame = frame * 0x27d4eb2d;
	frame = frame ^ (frame >> 15);

	return frame;
}

// 렌더링에 필요한 모든 자원들을 초기화하는 단계입니다.
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

// 화면을 생성하는 코드입니다.
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

	Throw(D3DCompileFromFile(shaderPath.c_str(), nullptr, nullptr, "vert", "vs_5_0", compileFlag, 0, &vBlob, &errBlob));
	Throw(D3DCompileFromFile(shaderPath.c_str(), nullptr, nullptr, "frag", "ps_5_0", compileFlag, 0, &pBlob, &errBlob));
	Throw(mDevice->CreateVertexShader(vBlob->GetBufferPointer(), vBlob->GetBufferSize(), nullptr, mScreenVS.GetAddressOf()));
	Throw(mDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), nullptr, mScreenPS.GetAddressOf()));
	Throw(mDevice->CreateInputLayout(inputElements, 2, vBlob->GetBufferPointer(), vBlob->GetBufferSize(), mScreenIL.GetAddressOf()));
}

// 화면을 그립니다.
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

		// 각 삼각형 인덱스에 해당하는 모든 정점들을 읽어옵니다.
		XMFLOAT4 position1 = XMFLOAT4(vertices[index1].mPosition.x, vertices[index1].mPosition.y, vertices[index1].mPosition.z, 1.0f);
		XMFLOAT4 position2 = XMFLOAT4(vertices[index2].mPosition.x, vertices[index2].mPosition.y, vertices[index2].mPosition.z, 1.0f);
		XMFLOAT4 position3 = XMFLOAT4(vertices[index3].mPosition.x, vertices[index3].mPosition.y, vertices[index3].mPosition.z, 1.0f);

		// pre-edge calculation
		// static mesh의 경우 edge는 한 번만 계산하면 되기 때문에
		// 미리 계산하여 kernel 호출 시의 GPU 연산 부하를 줄입니다.
		mMeshTriangles.push_back(XMFLOAT4(position1.x, position1.y, position1.z, 0.0f));
		mMeshTriangles.push_back(XMFLOAT4(position2.x - position1.x, position2.y - position1.y, position2.z - position1.z, 1.0f));
		mMeshTriangles.push_back(XMFLOAT4(position3.x - position2.x, position3.y - position2.y, position3.z - position2.z, 0.0f));
	}


	// 임의로 초기화된 바운딩 박스
	mMeshBoundingBox[0] = make_float3(1250.0f, 1250.0f, 1250.0f);
	mMeshBoundingBox[1] = make_float3(-1250.0f, -1250.0f, -1250.0f);

	long long triangleSize = indices.size() * sizeof(XMFLOAT3);
	uint triangleCount = indices.size() / 3;

	// 모든 삼각형 정보를 cudaTexture 1D 에 추가합니다.
	// 이 과정도 static mesh라면 한 번만 계산합니다.
	if (triangleSize > 0)
	{
		cudaError_t error = cudaMalloc(&cudaTriangles, triangleSize);
		GetError(error);


		error = cudaMemcpy(cudaTriangles, mMeshTriangles.data(), triangleSize, cudaMemcpyHostToDevice);
		GetError(error);

		InitTriangleTexture(cudaTriangles, triangleCount);

	}

	mTriangleCount = triangleCount;

	cudaSceneBoundBoxMax = mMeshBoundingBox[0];
	cudaSceneBoundBoxMin = mMeshBoundingBox[1];
}