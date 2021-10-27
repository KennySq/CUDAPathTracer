#include<cmath>
#include<cuda_runtime_api.h>
#include"math_functions.h"
#include"cuda_runtime.h"

#define PI 3.14159265359f

inline __host__ __device__ float3 absf3(float3 v)
{
	return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}

inline __host__ __device__ float InverseSqrt(float x)
{
	return 1.0f / sqrt(x);
}

inline __host__ __device__ float Dot(float3 v1, float3 v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline __host__ __device__ float3 operator+=(float3 v1, float3 v2)
{
	v1.x += v2.x;
	v1.y += v2.y;
	v1.z += v2.z;

	return v1;
}

inline __host__ __device__ float3 operator*(float3 v1, float3 v2)
{
	return make_float3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

inline __host__ __device__ float3 operator+(float3 v1, float3 v2)
{
	return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

inline __host__ __device__ float3 operator-(float3 v1, float3 v2)
{
	return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

inline __host__ __device__ float3 operator*(float3 v, float s)
{
	return make_float3(v.x * s, v.y * s, v.z * s);
}

inline __host__ __device__ float3 Normalize(float3 v)
{
	float len = InverseSqrt(Dot(v, v));

	return v * len;
}

inline __host__ __device__ float3 Cross(float3 v1, float3 v2)
{
	return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}