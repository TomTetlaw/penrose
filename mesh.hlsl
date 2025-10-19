
struct VertInput
{
  float3 position: POSITION;
  float2 tex_coord: TEXCOORD0;
  float3 normal: TEXCOORD2;
  float4 tangent: TEXCOORD3;
};

struct FragInput
{
  float4 clip_position: SV_Position;
  float3 frag_pos: TEXCOORD0;
  float2 tex_coord: TEXCOORD1;
  float3 view_dir: TEXCOORD2;
  float3 light_dir: TEXCOORD3;
  float4 material_params: TEXCOORD4;
  float4 colour: TEXCOORD5;
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
  float2 tex_coord = input.tex_coord;
  float4 world_position = mul(instance.transform, float4(input.position, 1));
  float4 view_position = mul(world_to_view, world_position);
  float4 clip_position = mul(view_to_clip, view_position);

  float3x3 adj = adjoint(instance.transform);
  float3 normal = normalize(mul(adj, input.normal));
  float3 tangent = normalize(mul(adj, input.tangent.xyz));
  tangent = normalize(tangent - dot(tangent, normal) * normal);
  float3 bitangent = normalize(cross(normal, tangent) * input.tangent.w);
  float3x3 tbn = float3x3(tangent, bitangent, normal);

  float3 view_dir = camera_pos - world_position.xyz;
  float3 light_dir = light_pos - world_position.xyz;

  FragInput output;
  output.clip_position = clip_position;
  output.tex_coord = tex_coord;
  output.frag_pos = mul(tbn, world_position.xyz);
  output.view_dir = mul(tbn, view_dir);
  output.light_dir = mul(tbn, light_dir);
  output.material_params = instance.material_params;
  output.colour = instance.colour;
  return output;
}

Texture2D texture0: register(t0, space2);
SamplerState sampler0: register(s0, space2);
Texture2D texture1: register(t1, space2);
SamplerState sampler1: register(s1, space2);
Texture2D texture2: register(t2, space2);
SamplerState sampler2: register(s2, space2);

enum LightType
{
  LightType_Directional = 0,
  LightType_Point = 1,
  LightType_Spot = 2,
};

struct LightInfo
{
  float3 position;
  LightType type;
  float3 colour;
  float brightness;
  float3 dir;
  float pad0;
  float inner_radius;
  float outer_radius;
  float2 pad1;
};

cbuffer FragConstantBuffer : register(b0, space3)
{
  LightInfo light;
}

void lighting_point(LightInfo light, float3 frag_pos, float3 l, float dist, float3 v, float3 n, out float diffuse, out float atten)
{
  atten = saturate((light.outer_radius - dist) / (light.outer_radius - light.inner_radius));
  atten *= atten;
  diffuse = max(dot(n, l), 0.0) * light.brightness * atten;
}

void lighting_spot(LightInfo light, float3 frag_pos, float3 l, float dist, float3 v, float3 n, out float diffuse, out float atten)
{
  float dist_atten = saturate(1.0 - (dist / light.outer_radius));
  dist_atten *= dist_atten;
  float inner_cos = cos(radians(light.inner_radius));
  float outer_cos = cos(radians(light.outer_radius));
  float cone_dot = dot(l, normalize(light.dir));
  float cone_atten = saturate((cone_dot - outer_cos) / (inner_cos - outer_cos));
  cone_atten = cone_atten * cone_atten;
  atten = dist_atten * cone_atten;
  diffuse = max(dot(n, l), 0.0) * light.brightness * atten;
}

void lighting(LightInfo light, float3 frag_pos, float3 l, float dist, float3 v, float3 n, float spec_intensity, float3 object_colour, inout float3 colour, float4 material_params)
{
  float atten;
  if (light.type == LightType_Point)
  {
    float diffuse;
    lighting_point(light, frag_pos, l, dist, v, n, diffuse, atten);
    colour += object_colour * light.colour * diffuse;
  }
  else if (light.type == LightType_Spot)
  {
    float diffuse;
    lighting_spot(light, frag_pos, l, dist, v, n, diffuse, atten);
    colour += object_colour * light.colour * diffuse;
  }
  float3 h = normalize(l + v);
  float spec = pow(max(0, dot(n, h)), material_params.z) * spec_intensity * light.brightness * atten;
  colour += light.colour * spec;
}

float4 frag_main(FragInput input) : SV_Target
{
  float2 tex_coord = input.tex_coord * input.material_params.xy;
  float4 colour_spec_map = texture0.Sample(sampler0, tex_coord);
  float3 normal_map = texture1.Sample(sampler1, tex_coord).rgb * 2.0 - 1.0;
  float normal_length = length(normal_map);
  float3 n = normalize(normal_map);
  float3 object_colour = input.colour.rgb * colour_spec_map.rgb;
  float3 colour = float3(0, 0, 0);
  float3 l = normalize(input.light_dir);
  float dist = length(input.light_dir);
  float3 v = normalize(input.view_dir);
  float spec = texture2.Sample(sampler2, tex_coord).r;
  lighting(light, input.frag_pos, l, dist, v, n, spec, object_colour, colour, input.material_params);
  float4 result = float4(colour, 1);
  return result;
}