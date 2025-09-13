
struct FragInput
{
  float4 clip_position: SV_Position;
  float2 tex_coord: TEXCOORD0;
  float4 colour: TEXCOORD1;
};

cbuffer ConstantBuffer : register(b0, space1)
{
  float4x4 screen_to_clip;
};

struct InstanceData
{
	float4 position_size;
	float4 colour;
};

StructuredBuffer<InstanceData> instance_buffer: register(t0, space0);

static const float2 positions[4] =
{
  float2(0, 0),
  float2(1, 0),
  float2(0, 1),
  float2(1, 1),
};

static const float2 tex_coords[4] =
{
  float2(0, 0),
  float2(1, 0),
  float2(0, 1),
  float2(1, 1),
};

FragInput vert_main(uint vertex_id: SV_VertexId, uint instance_id: SV_InstanceId)
{
  InstanceData instance = instance_buffer[instance_id];
  float2 position = positions[vertex_id % 4];
  float2 screen_position = instance.position_size.xy + position*instance.position_size.zw;
  float2 tex_coord = tex_coords[vertex_id % 4];
  float4 clip_position = mul(float4(screen_position, 0, 1), screen_to_clip);
  FragInput output;
  output.clip_position = clip_position;
  output.tex_coord = tex_coord;
  output.colour = instance.colour;
  return output;
}

float4 frag_main(FragInput input): SV_Target
{
  return input.colour;
}