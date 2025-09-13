
struct VertexInput
{
	float3 position: POSITION;
};

struct FragInput
{
  float4 clip_position: SV_Position;
  float4 colour: TEXCOORD0;
};

cbuffer ConstantBuffer : register(b0, space1)
{
  float4x4 world_to_view;
  float4x4 view_to_clip;
};

struct InstanceData
{
	float4x4 local_to_world;
	float4 colour;
};

StructuredBuffer<InstanceData> instance_buffer: register(t0, space0);

FragInput vert_main(VertexInput input, uint instance_id: SV_InstanceId)
{
  InstanceData instance = instance_buffer[instance_id];
  float4x4 local_to_world = instance.local_to_world;
  float4 colour = instance.colour;
  float4 world_position = mul(float4(input.position, 1), local_to_world);
  float4 view_position = mul(world_position, world_to_view);
  float4 clip_position = mul(view_position, view_to_clip);
  FragInput output;
  output.clip_position = clip_position;
  output.colour = colour;
  return output;
}

float4 frag_main(FragInput input): SV_Target
{
  return input.colour;
}