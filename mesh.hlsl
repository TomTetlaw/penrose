
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
  float3 world_pos: TEXCOORD0;
  float2 tex_coord: TEXCOORD1;
  float4 colour: TEXCOORD2;
  float3 tangent: TEXCOORD3;
  float3 view_dir: TEXCOORD4;
  float3 light_dir: TEXCOORD5;
};

cbuffer VertConstantBuffer : register(b0, space1)
{
  row_major float4x4 world_to_view;
  row_major float4x4 view_to_clip;
  float3 camera_pos;
  float pad0;
  float4 viewport;
  float3 sun_dir;
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
  float3x3 tbn = float3x3(tangent, bitangent, normal);
  float3 view_dir = normalize(mul(tbn, normalize(camera_pos - world_position.xyz)));
  float3 light_dir = normalize(mul(tbn, sun_dir));
  FragInput output;
  output.clip_position = clip_position;
  output.world_pos = world_position.xyz;
  output.tex_coord = tex_coord;
  output.colour = colour;
  output.tangent = tangent;
  output.view_dir = view_dir;
  output.light_dir = light_dir;
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

struct DirectionalLight
{
  float3 dir;
  float brightness;
  float3 colour;
};

struct PointLight
{
  float3 position;
  float radius;
  float brightness;
  float3 colour;
};

struct SpotLight
{
  float3 position;
  float3 dir;
  float inner_radius;
  float outer_radius;
  float brightness;
  float3 colour;
};

float2 lighting_directional(DirectionalLight light, float3 n)
{
    float diffuse = max(dot(n, light.dir), 0.0) * light.brightness;
    return float2(diffuse, spec);
}

void lighting_point(PointLight light, float3 world_pos, float3 v, float3 n, out float diffuse)
{
  float3 L = light.position - world_pos;
  float dist = length(L);
  L /= dist;
  float atten = saturate(1.0 - (dist / light.radius));
  atten = atten * atten;
  diffuse = max(dot(n, L), 0.0);
  float3 h = normalize(L + v);
  diffuse *= light.brightness * atten;
}

void lighting_spot(SpotLight light, float3 world_pos, float3 v, float3 n, out float diffuse)
{
  float3 L = light.position - world_pos;
  float dist = length(L);
  L /= dist;
  float dist_atten = saturate(1.0 - (dist / light.outer_radius));
  dist_atten = dist_atten * dist_atten;
  float cone_dot = dot(-L, normalize(light.dir));
  float cone_atten = pow(saturate(cone_dot), 3);
  float total_atten = dist_atten * cone_atten;
  diffuse = max(dot(n, L), 0.0);
  diffuse *= light.brightness * total_atten;
}

float4 frag_main(FragInput input) : SV_Target
{
  float2 tex_coord = input.tex_coord * tex_coord_scale;
  float3 map_colour = texture0.Sample(sampler0, tex_coord).rgb;
  float3 colour = input.colour.rgb * map_colour;
  float3 v = normalize(input.view_dir);
  float3 n = normalize((texture1.Sample(sampler1, tex_coord).rgb * 2.0 - 1.0) * float3(1, 1, 1));
  float3 l = normalize(-input.light_dir);

  DirectionalLight directional_light;
  directional_light.dir = l;
  directional_light.brightness = 1.0;
  directional_light.colour = float3(1, 1, 1);
  float diffuse;
  lighting_directional(directional_light, n, diffuse);

  PointLight point_light;
  point_light.position = float3(-101, -16, 74);
  point_light.radius = 15;
  point_light.brightness = 1;
  point_light.colour = float3(1, 0, 0);
  float point_diffuse;
  lighting_point(point_light, input.world_pos, v, n, point_diffuse);

  SpotLight spot_light;
  spot_light.position = float3(-101, -16, 75);
  spot_light.dir = float3(0, -1, 0);
  spot_light.inner_radius = 5;
  spot_light.outer_radius = 10.0;
  spot_light.brightness = 5.0;
  spot_light.colour = float3(0, 0, 1);
  float spot_diffuse;
  lighting_spot(spot_light, input.world_pos, v, n, spot_diffuse);

  float3 spec_map = texture2.Sample(sampler2, tex_coord).rgb;
  colour *= lighting_intensity * diffuse;
  colour += point_diffuse * point_light.colour;
  colour += spot_diffuse * spot_light.colour;
  colour += lighting_intensity * spec * specular_intensity * spec_map;
  float4 result = float4(colour * input.colour.a, input.colour.a);
  return result;
}