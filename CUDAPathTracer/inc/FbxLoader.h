#pragma once
#include<fbxsdk.h>
#include<d3d11.h>
#include<DirectXMath.h>
#include<vector>
#include<unordered_map>

#pragma comment(lib, "libfbxsdk-md.lib")
#pragma comment(lib, "libxml2-md.lib")
#pragma comment(lib, "zlib-md.lib")

#pragma comment(lib, "d3d11.lib")

using namespace DirectX;

struct Vertex
{
	XMFLOAT3 mPosition;
	XMFLOAT3 mNormal;
	XMFLOAT3 mBinormal;
	XMFLOAT3 mTangent;
	XMFLOAT2 mTexcoord;
	unsigned int mMaterialID;

	bool operator==(const Vertex& v2) const
	{
		bool bPosition = (mPosition.x == v2.mPosition.x) && (mPosition.y == v2.mPosition.y) && (mPosition.z == v2.mPosition.z);
		bool bNormal = (mNormal.x == v2.mNormal.x) && (mNormal.y == v2.mNormal.y) && (mNormal.z == v2.mNormal.z);
		bool bBinormal = (mBinormal.x == v2.mBinormal.x) && (mBinormal.y == v2.mBinormal.y) && (mBinormal.z == v2.mBinormal.z);
		bool bTangent = (mTangent.x == v2.mTangent.x) && (mTangent.y == v2.mTangent.y) && (mTangent.z == v2.mTangent.z);
		bool bTexcoord = (mTexcoord.x == v2.mTexcoord.x) && (mTexcoord.y == v2.mTexcoord.y);

		return bPosition && bNormal && bBinormal && bTangent && bTexcoord;

	}
};

namespace std
{
	template<>
	class std::hash<Vertex>
	{
	public:
		size_t operator()(const Vertex& v) const
		{
			using std::hash;

			hash<float> h;

			size_t posx = h(v.mPosition.x);
			size_t posy = h(v.mPosition.y);
			size_t posz = h(v.mPosition.z);

			size_t normx = h(v.mNormal.x);
			size_t normy = h(v.mNormal.y);
			size_t normz = h(v.mNormal.z);

			size_t binormx = h(v.mBinormal.x);
			size_t binormy = h(v.mBinormal.y);
			size_t binormz = h(v.mBinormal.z);

			size_t tanx = h(v.mTangent.x);
			size_t tany = h(v.mTangent.y);
			size_t tanz = h(v.mTangent.z);

			size_t uvx = h(v.mTexcoord.x);
			size_t uvy = h(v.mTexcoord.y);

			return (posx ^ posy ^ posz ^ normx ^ normy ^ normz ^ binormx ^ binormy ^ binormz ^ tanx ^ tany ^ tanz ^ uvx ^ uvy);
		}
	};
}




class FbxLoader
{
public:
	FbxLoader(const char* path) noexcept;
	virtual ~FbxLoader() {}

	std::vector<Vertex> Vertices;
	std::vector<unsigned int> Indices;

protected:
	bool fbx_loadFbx();
	void fbx_loadNode(FbxNode* node);
	void fbx_getControlPoints(FbxMesh* mesh);
	void fbx_toVertex(const XMFLOAT3& position, const XMFLOAT3& normal, const XMFLOAT3& binormal, const XMFLOAT3& tangent, const XMFLOAT2& texcoord);
	XMFLOAT3 fbx_getNormal(const FbxMesh* mesh, int controlPoint, int vertex);
	XMFLOAT3 fbx_getBinormals(const FbxMesh* mesh, int controlPoint, int vertex);
	XMFLOAT3 fbx_getTangents(const FbxMesh* mesh, int controlPoint, int vertex);
	XMFLOAT2 fbx_getTexcoords(const FbxMesh* mesh, int controlPoint, int texcoordIndex);
	unsigned int fbx_getMaterialID(const FbxMesh* mesh, int polygon);
	void fbx_insertVertex(const XMFLOAT3& position, const XMFLOAT3& normal, const XMFLOAT3& binormal, const XMFLOAT3& tangent, const XMFLOAT2& uv, int materialID, unsigned int vertexCount);



	std::string mPath;

	std::vector<XMFLOAT3> mPositions;
	std::vector<XMFLOAT3> mNormals;
	std::vector<XMFLOAT3> mBinormals;
	std::vector<XMFLOAT3> mTangents;
	std::vector<XMFLOAT2> mTexcoords;
	std::vector<unsigned int> mMaterialIDs;

	unsigned int mMeshCount = 0;
	unsigned int mTotalTriangles = 0;
	unsigned int mTotalVertices = 0;
	unsigned int mIndexCount = 0;

	std::unordered_map<Vertex, unsigned int> mIndexMap;
};


