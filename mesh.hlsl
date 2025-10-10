
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
  float3 tangent: TEXCOORD2;
  float3 view_dir: TEXCOORD3;
  float3 light_dir: TEXCOORD4;
};

cbuffer VertConstantBuffer : register(b0, space1)
{
  row_major float4x4 world_to_view;
  row_major float4x4 view_to_clip;
  float3 camera_pos;
  float pad0;
}

struct InstanceData
{
  row_major float4x4 transform;
  float4 colour;
};

StructuredBuffer<InstanceData> instance_buffer: register(t0, space0);

float3x3 adjoint(float4x4 m)
{
  float3x3 result = float3x3
  (
    cross(m[1].xyz, m[2].xyz),
    cross(m[2].xyz, m[0].xyz),
    cross(m[0].xyz, m[1].xyz)
  );
  return result;
}

FragInput vert_main(VertInput input, int instance_id: SV_InstanceId)
{
  InstanceData instance = instance_buffer[instance_id];
  float4 colour = input.colour * instance.colour;
  float2 tex_coord = input.tex_coord;
  float4 world_position = mul(instance.transform, float4(input.position, 1));
  float4 view_position = mul(world_to_view, world_position);
  float4 clip_position = mul(view_to_clip, view_position);
  float3x3 adj = adjoint(instance.transform);
  float3 normal = normalize(mul(adj, input.normal));
  float3 tangent = normalize(mul(adj, input.tangent));
  float3 bitangent = normalize(cross(normal, tangent));
  tangent = normalize(tangent - dot(tangent, normal) * normal);
  bitangent = cross(normal, tangent);
  float3x3 tbn = float3x3(tangent, bitangent, normal);
  float3 view_dir = mul(tbn, normalize(camera_pos - world_position.xyz));
  FragInput output;
  output.clip_position = clip_position;
  output.tex_coord = tex_coord;
  output.colour = colour;
  output.tangent = tangent;
  output.view_dir = view_dir;
  output.light_dir = mul(tbn, normalize(float3(0.5, 0.5, 1.0)));
  return output;
}

Texture2D texture0: register(t0, space2);
SamplerState sampler0: register(s0, space2);
Texture2D texture1: register(t1, space2);
SamplerState sampler1: register(s1, space2);
Texture2D texture2: register(t2, space2);
SamplerState sampler2: register(s2, space2);

cbuffer FragConstantBuffer : register(b0, space3)
{
  float specular_shininess;
  float specular_intensity;
  float2 tex_coord_scale;
  float lighting_intensity;
  float3 pad;
}

float4 frag_main(FragInput input) : SV_Target
{
  float2 tex_coord = input.tex_coord * tex_coord_scale;
  float3 colour = input.colour.rgb * texture0.Sample(sampler0, tex_coord).rgb;
  if (lighting_intensity > 0)
  {
    float3 v = normalize(input.view_dir);
    float3 n = normalize((texture1.Sample(sampler1, tex_coord).xyz * 2.0 - 1.0) * float3(1, -1, 1));
    float3 l = normalize(input.light_dir);
    float diffuse = max(dot(n, l), 0);
    float3 h = normalize(l + v);
    float specular = pow(max(dot(n, h), 0.0), specular_shininess);
    float3 spec_colour = specular*specular_intensity*texture2.Sample(sampler2, tex_coord).rgb;
    colour *= lighting_intensity * diffuse;
    colour += lighting_intensity * spec_colour;
  }
  return float4(colour * input.colour.a, input.colour.a);
}