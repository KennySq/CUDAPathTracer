#include"inc/PathTracer.cuh"

texture<float4, 1, cudaReadModeElementType> gCudaTriangleTexture;

float3 cudaSceneBoxMin;
float3 cudaSceneBoxMax;

float4* cudaRenderTexture;

struct Ray
{
	__device__ __host__ Ray(float3 origin, float3 direction)
		: Origin(origin), Direction(direction)
	{

	}

	float3 Origin;
	float3 Direction;
};

struct Sphere
{
	float Radius;
	float3 Position, Emission, Color;

	__device__ float Intersect(const Ray& r) const
	{
		float3 dir = Position - r.Origin;
		float a = Dot(r.Direction, r.Direction);
		float b = 2.0f * Dot(dir, r.Origin - Position);
		float c = Dot(Position, Position) + Dot(r.Origin, r.Origin) + -2.0f * Dot(Position, r.Origin) - (Radius * Radius);

		float disc = b * b - 4 * a * c;

		if (disc < 0.0f)
		{
			return 0.0f;
		}
		disc = sqrtf(disc);
		float t = (-b - disc) / (2.0f * a);

		return t;
	}
};

struct Plane
{
	float3 Center;
	float3 Normal;
	float Length;

	__device__ bool Intersect(const Ray& r) const
	{
		float denom = Dot(Normal, r.Direction);

		if (denom > 1e-6)
		{
			float numer = Dot(Center - r.Origin, Normal);
			float t = (numer / denom);
			
			return t > 1e-6;
		}

		return false;
	}

};

__constant__ Sphere Spheres[] =
{
	{0.25f, {0,0,0}, {0,0,0}, {1,0,0} },
};

__constant__ Plane Planes[] =
{
	{{0.0f, 0.0f, 10.0f},{0.0f, 0.0f, 1.0f }, 1.0f},
};

void PathTracer::Awake()
{
	loadAssets();

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

	cudaDeviceSynchronize();


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
	CUresult cuResult = cuSurfObjectCreate(&mCudaRenderSurface, &cuReourceDesc);


	//std::string meshPath = GetWorkingDirectoryA();
	//meshPath += "..\\..\\CUDAPathTracer\\resources\\shiba\\shiba.fbx";
	//FbxLoader loader(meshPath.c_str());

}

void PathTracer::Update(float delta)
{
	mContext->ClearRenderTargetView(mBackBufferRTV.Get(), Colors::Green);

}

__global__ void KernelIntersectScene(CUsurfObject surface, unsigned int width, unsigned int height)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	float3 origin = make_float3(0, 0, -10);
	float3 rayDir = make_float3(0, 0, 1);
	float4 color = make_float4(0, 0, 0, 0);

	Ray r = Ray(origin, rayDir);
	float3 uv = make_float3(x / (float)width, y / (float)height, 0.0f);

	float3 cx = make_float3(width * 1.5708f / (float)height, 0, 0);
	float3 cy = Normalize(Cross(cx, rayDir)) * 1.5708f;

	uv.x *= 1.777f;

	float3 dir = (cx * ((.25f + x) / (float)width - .5f) + cy * ((.25f + y) / (float)height - .5f)) + r.Direction;
	//float3 dir = Normalize(r.Direction + uv);

	r.Direction = Normalize(dir);

	color = make_float4(r.Direction.x, r.Direction.y, r.Direction.z, 1.0f);

	bool bIntersect = Planes[0].Intersect(r);
	if (bIntersect)
	{
		color = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
	}

	surf2Dwrite<float4>(color, surface, x * sizeof(float4), y);
}

void PathTracer::Render(float delta)
{
	dim3 block = dim3(16, 16, 1);
	dim3 grid = dim3(mWidth / block.y, mHeight / block.y, 1);

	KernelIntersectScene << <grid, block >> > (mCudaRenderSurface, mWidth, mHeight);
	drawScreen();

	cudaDeviceSynchronize();


	mSwapchain->Present(0, 0);

	mFrameIndex++;
}

void PathTracer::Release()
{
}

void PathTracer::drawScreen()
{
	static ID3D11ShaderResourceView* nullSrv[] = { nullptr };
	static ID3D11RenderTargetView* nullRtv[] = { nullptr };
	static uint strides[] = { sizeof(ScreenVertex) };
	static uint offsets[] = { 0 };

	mContext->IASetVertexBuffers(0, 1, mScreenVB.GetAddressOf(), strides, offsets);

	mContext->IASetIndexBuffer(mScreenIB.Get(), DXGI_FORMAT_R32_UINT, 0);
	mContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	mContext->IASetInputLayout(mScreenIL.Get());

	mContext->OMSetRenderTargets(1, mBackBufferRTV.GetAddressOf(), nullptr);

	mContext->VSSetShader(mScreenVS.Get(), nullptr, 0);
	mContext->PSSetShader(mScreenPS.Get(), nullptr, 0);
	mContext->PSSetShaderResources(0, 1, mRenderTextureSRV.GetAddressOf());

	mContext->RSSetViewports(1, &mViewport);

	mContext->DrawIndexed(6, 0, 0);

	mContext->OMSetRenderTargets(1, nullRtv, nullptr);

	mContext->PSSetShaderResources(0, 1, nullSrv);
}

void PathTracer::loadAssets()
{
	AcquireHardware();
	AllocConsole();

	HRESULT result = mSwapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(mBackBuffer.GetAddressOf()));
	assert(result == S_OK);

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
	D3D11_SUBRESOURCE_DATA subData{};

	subData.pSysMem = vertices;
	vbDesc.ByteWidth = sizeof(vertices);
	vbDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;

	result = mDevice->CreateBuffer(&vbDesc, &subData, mScreenVB.GetAddressOf());
	assert(result == S_OK);

	subData.pSysMem = indices;
	ibDesc.ByteWidth = sizeof(indices);
	ibDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;

	result = mDevice->CreateBuffer(&ibDesc, &subData, mScreenIB.GetAddressOf());
	assert(result == S_OK);


	D3D11_INPUT_ELEMENT_DESC inputElements[] =
	{
		{"POSITION",0, DXGI_FORMAT_R32G32B32_FLOAT, 0,0, D3D11_INPUT_PER_VERTEX_DATA, 0},
		{"TEXCOORD",0, DXGI_FORMAT_R32G32_FLOAT, 0,0xFFFFFFFF, D3D11_INPUT_PER_VERTEX_DATA, 0},
	};

	ID3DBlob* vBlob, * pBlob, * errBlob;

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
