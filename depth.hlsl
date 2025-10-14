
struct VertInput
{
  float3 position: POSITION;
};

struct FragInput
{
  float4 clip_position: SV_Position;
};

cbuffer VertConstantBuffer : register(b0, space1)
{
  row_major float4x4 world_to_view;
  row_major float4x4 view_to_clip;
  float3 camera_pos;
  float pad0;
  float4 viewport;
  float3 light_pos;
  float pad1;
}

struct InstanceData
{
  row_major float4x4 transform;
  float4 colour;
  float4 material_params;
};

StructuredBuffer<InstanceData> instance_buffer: register(t0, space0);

FragInput vert_main(VertInput input, int instance_id: SV_InstanceId)
{
  InstanceData instance = instance_buffer[instance_id];
  float4 world_position = mul(instance.transform, float4(input.position, 1));
  float4 view_position = mul(world_to_view, world_position);
  float4 clip_position = mul(view_to_clip, view_position);
  FragInput output;
  output.clip_position = clip_position;
  return output;
}

float4 frag_main(FragInput input) : SV_Target
{
  return float4(0, 0, 0, 0);
}