
struct VertInput
{
  float3 position: POSITION;
  float2 tex_coord: TEXCOORD0;
  float3 normal: TEXCOORD1;
  float3 tangent: TEXCOORD2;
};

struct FragInput
{
  float4 clip_position: SV_Position;
  float4 colour: TEXCOORD0;
  float2 tex_coord: TEXCOORD1;
  float4 props: TEXCOORD2;
};

cbuffer VertConstantBuffer : register(b0, space1)
{
  row_major float4x4 world_to_view;
  row_major float4x4 view_to_clip;
  float3 camera_pos;
  float pad0;
  float4 viewport;
}

struct InstanceData
{
  float3 position;
  float pad0;
  float4 colour;
  float4 props;
};

StructuredBuffer<InstanceData> instance_buffer: register(t0, space0);

FragInput vert_main(VertInput input, int instance_id: SV_InstanceId)
{
  InstanceData instance = instance_buffer[instance_id];
  float4 colour = instance.colour;
  float2 tex_coord = input.tex_coord;
  float2 size = instance.props.xy;
  float4 view_center = mul(world_to_view, float4(instance.position, 1));
  float4 clip_center = mul(view_to_clip, view_center);
  float2 clip_size = size / viewport.xy * 2.0;
  float2 corner_offset = input.position.xy * clip_size;
  float4 clip = clip_center;
  clip.xy += corner_offset * clip.w;
  FragInput output;
  output.clip_position = clip;
  output.tex_coord = tex_coord;
  output.colour = colour;
  output.props = instance.props;
  return output;
}

float4 frag_main(FragInput input) : SV_Target
{
  float2 quad_size = input.props.xy;
  float border_thickness = input.props.z;
  if (border_thickness > 0.0)
  {
    float2 pos_px = (input.tex_coord - 0.5) * quad_size;
    float2 d = (quad_size * 0.5) - abs(pos_px);
    float dist = min(d.x, d.y);
    if (dist < 0 || dist > border_thickness)
        discard;
  }
  float3 colour = input.colour.rgb;
  float alpha = input.colour.a;
  return float4(colour * alpha, alpha);
}