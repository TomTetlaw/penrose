
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
  float3 frag_pos: TEXCOORD0;
  float2 tex_coord: TEXCOORD1;
  float4 colour: TEXCOORD2;
  float3 view_dir: TEXCOORD3;
  float3 light_pos: TEXCOORD4;
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
  bitangent = normalize(cross(normal, tangent));
  float3x3 tbn = transpose(float3x3(tangent, bitangent, normal));
  float3 view_dir = normalize(mul(tbn, normalize(camera_pos - world_position.xyz)));
  FragInput output;
  output.clip_position = clip_position;
  output.tex_coord = tex_coord;
  output.colour = colour;
  output.frag_pos = mul(tbn, world_position.xyz);
  output.view_dir = view_dir;
  output.light_pos =  mul(tbn, light_pos);
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
  float radius;
  float inner_radius;
  float outer_radius;
  float2 pad;
};

cbuffer FragConstantBuffer : register(b0, space3)
{
  float specular_shininess;
  float3 pad2;
  float2 tex_coord_scale;
  float2 pad3;
  LightInfo light;
}

void lighting_directional(LightInfo light, float3 L, float dist, float3 v, float3 n, out float diffuse, out float spec)
{
    diffuse = max(dot(n, -light.dir), 0.0) * light.brightness;
    float3 h = normalize(-light.dir + v);
    spec = pow(max(0, dot(n, h)), specular_shininess);
}

void lighting_point(LightInfo light, float3 frag_pos, float3 L, float dist, float3 v, float3 n, out float diffuse, out float spec)
{
  float atten = saturate(1.0 - (dist / light.radius));
  atten *= atten;
  diffuse = max(dot(n, L), 0.0) * light.brightness * atten;
  float3 h = normalize(L + v);
  spec = pow(max(0, dot(n, h)), specular_shininess) * light.brightness * atten;
}

void lighting_spot(LightInfo light, float3 frag_pos, float3 L, float dist, float3 v, float3 n, out float diffuse, out float spec)
{
  float dist_atten = saturate(1.0 - (dist / light.outer_radius));
  dist_atten *= dist_atten;
  float inner_cos = cos(radians(light.inner_radius));
  float outer_cos = cos(radians(light.outer_radius));
  float cone_dot = dot(L, normalize(light.dir));
  float cone_atten = saturate((cone_dot - outer_cos) / (inner_cos - outer_cos));
  cone_atten = cone_atten * cone_atten;
  float total_atten = dist_atten * cone_atten;
  diffuse = max(dot(n, L), 0.0) * light.brightness * total_atten;
  float3 h = normalize(L + v);
  spec = pow(max(0, dot(n, h)), specular_shininess) * light.brightness * total_atten;
}

void lighting(LightInfo light, float3 frag_pos, float3 L, float dist, float3 v, float3 n, float spec_map, float3 object_colour, inout float3 colour)
{
  float diffuse, spec;
  if (light.type == LightType_Directional)
  {
    lighting_directional(light, L, dist, v, n, diffuse, spec);
    colour += object_colour * light.colour * diffuse;
  }
  else if (light.type == LightType_Point)
  {
    lighting_point(light, frag_pos, L, dist, v, n, diffuse, spec);
    colour += object_colour * light.colour * diffuse;
  }
  else if (light.type == LightType_Spot)
  {
    lighting_spot(light, frag_pos, L, dist, v, n, diffuse, spec);
    colour += object_colour * light.colour * diffuse;
  }
  colour += light.colour * spec * spec_map * 2;
}

float4 frag_main(FragInput input) : SV_Target
{
  float2 tex_coord = input.tex_coord * tex_coord_scale;

  float3 v = normalize(input.view_dir);
  float3 n = normalize((texture1.Sample(sampler1, tex_coord).rgb * 2.0 - 1.0) * float3(1, 1, 1));

  float3 object_colour = input.colour.rgb * texture0.Sample(sampler0, tex_coord).rgb;
  float3 colour = float3(0, 0, 0);
  float spec_map = texture2.Sample(sampler2, tex_coord).r;
  float3 light_dir = input.light_pos - input.frag_pos;
  float3 L = normalize(light_dir);
  float dist = length(light_dir);
  lighting(light, input.frag_pos, L, dist, v, n, spec_map, object_colour, colour);

  float4 result = float4(colour * input.colour.a, input.colour.a);
  return result;
}