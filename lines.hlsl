
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
  float2 tex_coord: TEXCOORD0;
  float4 colour: TEXCOORD1;
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
  float3 start;
  float pad0;
  float3 end;
  float pad1;
  float4 colour;
  float4 props;
};

StructuredBuffer<InstanceData> instance_buffer: register(t0, space0);

void trim_line(inout float4 start, inout float4 end)
{
  float n = viewport.z;
  if (start.x < n && end.x < n)
  {
		return;
	}
	float t = (n - start.x) / (end.x - start.x);
	if (start.x < n)
	{
		start = lerp(start, end, t);
	}
	if (end.x < n)
	{
		end = lerp(start, end, t);
	}
}

FragInput vert_main(VertInput input, int instance_id: SV_InstanceId)
{
  InstanceData instance = instance_buffer[instance_id];
  float thickness = instance.props.x;
  FragInput output;
  float4 colour = instance.colour;
  float2 tex_coord = input.tex_coord;
  float3 start = instance.start;
  float3 end = instance.end;
  float3 p = input.position;
  float4 view_start = mul(world_to_view, float4(start, 1.0));
  float4 view_end = mul(world_to_view, float4(end, 1.0));
  trim_line(view_start, view_end);
  trim_line(view_end, view_start);
  float4 clip_start = mul(view_to_clip, view_start);
  float4 clip_end = mul(view_to_clip, view_end);
  float2 ndc_start = clip_start.xy / clip_start.w;
  float2 ndc_end = clip_end.xy / clip_end.w;
  float2 dir = ndc_end - ndc_start;
  float aspect = viewport.x / viewport.y;
  dir.x *= aspect;
  dir = normalize(dir);
  float2 offset = float2(dir.y, -dir.x);
  dir.x /= aspect;
  offset.x /= aspect;
  if (p.x < 0)
  {
    offset *= -1.0;
  }
  if (p.y < 0.0)
  {
    offset -= dir;
  }
  if (p.y > 1.0)
  {
    offset += dir;
  }
  offset *= thickness;
  offset /= viewport.y;
  float4 clip = p.y < 0.5 ? clip_start : clip_end;
  offset *= clip.w;
  clip.xy += offset;
  output.clip_position = clip;
  output.tex_coord = tex_coord;
  output.colour = colour;
  return output;
}

float4 frag_main(FragInput input) : SV_Target
{
  float2 tex_coord = input.tex_coord;
  if (abs(tex_coord.y) > 1.0)
  {
    float a = tex_coord.x;
    float b = (tex_coord.y > 0.0) ? tex_coord.y - 1.0 : tex_coord.y + 1.0;
    float len = a * a + b * b;
    if (len > .5)
    {
      discard;
    }
  }
  float4 colour = input.colour;
  return float4(colour.rgb * colour.a, colour.a);
}