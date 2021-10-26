SamplerState defaultSampler : register(s0);
Texture2D tex : register(t0);

struct VertexInput
{
    float3 Position : POSITION0;
    float2 Texcoord : TEXCOORD0;
};

struct PixelInput
{
    float4 Position : SV_Position;
    float2 Texcoord : TEXCOORD0;
};


PixelInput vert(VertexInput input) 
{
    PixelInput output = (PixelInput)0;
    
    output.Position = float4(input.Position, 1.0f);
    output.Texcoord = input.Texcoord;
    
	return output;
}

float4 frag(PixelInput input) : SV_Target0
{
    return tex.Sample(defaultSampler, input.Texcoord);

}