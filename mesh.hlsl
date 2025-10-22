
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
  LightType_Ambient = 0,
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

float3 lighting_ggx(float3 N, float3 V, float3 L, float3 albedo, float3 aorm, float3 light_colour, float4 material_params)
{
    #define PI 3.14159265
    float3 H = normalize(V + L);
    float NdotL = saturate(dot(N, L));
    float NdotV = saturate(dot(N, V));
    float NdotH = saturate(dot(N, H));
    float VdotH = saturate(dot(V, H));
    NdotV = max(NdotV, 0.001);
    NdotL = max(NdotL, 0.001);
    float metallic = saturate(aorm.b * material_params.g);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
    float roughness = aorm.g * material_params.r;
    roughness = roughness * roughness;
    float3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    float D = a2 / max(PI * denom * denom, 1e-5);
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float Gv = NdotV / (NdotV * (1.0 - k) + k);
    float Gl = NdotL / (NdotL * (1.0 - k) + k);
    float G = Gv * Gl;
    float3 specular_brdf = (D * G * F) / (4.0 * NdotV * NdotL + 1e-5);
    float3 kS = F;
    float3 kD = (1.0 - kS) * (1.0 - metallic);
    float3 diffuse_brdf = kD * albedo / PI;
    float ao = aorm.r;
    diffuse_brdf *= ao;
    float3 radiance = light_colour * NdotL;
    return (diffuse_brdf + specular_brdf) * radiance;
}

float3 lighting(LightInfo light, float3 frag_pos, float3 l, float dist, float3 v, float3 n, float3 aorm, float3 object_colour, float4 material_params)
{
  if (light.type == LightType_Ambient)
  {
    return lighting_ggx(n, v, l, object_colour, aorm, .001, material_params);
  }
  else if (light.type == LightType_Point)
  {
    float atten = saturate((light.outer_radius - dist) / (light.outer_radius - light.inner_radius));
    atten *= atten;
    return lighting_ggx(n, v, l, object_colour, aorm, light.colour * light.brightness * atten, material_params);
  }
  else
  {
    return float3(0, 0, 0);
  }
}

float4 frag_main(FragInput input) : SV_Target
{
  float2 tex_coord = input.tex_coord * input.material_params.zw;
  float3 colour_map = texture0.Sample(sampler0, tex_coord).rgb;
  float3 normal_map = texture1.Sample(sampler1, tex_coord).rgb * 2.0 - 1.0;
  float3 aorm_map = texture2.Sample(sampler2, tex_coord).rgb;
  float normal_length = length(normal_map);
  float3 n = normalize(normal_map);
  float3 object_colour = input.colour.rgb * colour_map.rgb;
  float3 l = normalize(input.light_dir);
  float dist = length(input.light_dir);
  float3 v = normalize(input.view_dir);
  float3 colour = lighting(light, input.frag_pos, l, dist, v, n, aorm_map, object_colour, input.material_params);
  colour = max(colour, 0.0);
  colour = pow(colour, 1.0 / 2.2);
  float4 result = float4(colour, 1);
  return saturate(result);
}