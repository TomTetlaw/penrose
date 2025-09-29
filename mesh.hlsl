
struct VertInput
{
  float3 position: POSITION;
  float2 tex_coord: TEXCOORD0;
  float4 colour: TEXCOORD1;
  float3 normal: TEXCOORD2;
  float3 tangent: TEXCOORD3;
};

struct FragInput
{
  float4 clip_position: SV_Position;
  float2 tex_coord: TEXCOORD0;
  float4 colour: TEXCOORD1;
  float3 normal: TEXCOORD2;
  float3 tangent: TEXCOORD3;
};

cbuffer ConstantBuffer : register(b0, space1)
{
  row_major float4x4 world_to_view;
  row_major float4x4 view_to_clip;
}

struct InstanceData
{
  row_major float4x4 transform;
  float4 colour;
};

StructuredBuffer<InstanceData> instance_buffer: register(t0, space0);

float3x3 adjoint(float4x4 m)
{
  float3x3 m3 = float3x3
  (
    cross(m[1].xyz, m[2].xyz),
    cross(m[2].xyz, m[0].xyz),
    cross(m[0].xyz, m[1].xyz)
  );
  float3x3 result = transpose(m3);
  return result;
}

FragInput vert_main(VertInput input, int instance_id: SV_InstanceId) {
  InstanceData instance = instance_buffer[instance_id];
  float4 colour = input.colour * instance.colour;
  float2 tex_coord = input.tex_coord;
  float4 world_position = mul(instance.transform, float4(input.position, 1));
  float4 view_position = mul(world_to_view, world_position);
  float4 clip_position = mul(view_to_clip, view_position);
  float3x3 adj = adjoint(instance.transform);
  float3 normal = normalize(mul(adj, input.normal));
  float3 tangent = normalize(mul(adj, input.tangent));
  FragInput output;
  output.clip_position = clip_position;
  output.tex_coord = tex_coord;
  output.colour = colour;
  output.normal = normal;
  output.tangent = tangent;
  return output;
}

float4 frag_main(FragInput input) : SV_Target
{
  float alpha = input.colour.a;
  float3 colour = input.colour.rgb;
  return float4(colour * alpha, alpha);
}