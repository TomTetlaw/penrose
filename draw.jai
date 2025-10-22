
dr_shaders_dir :: "shaders";
dr_max_mesh_instances :: 10240;
dr_max_line_instances :: 10240;
dr_max_quad_instances :: 10240;
dr_colour_format :: SDL_GPUTextureFormat.B8G8R8A8_UNORM_SRGB;
dr_depth_format :: SDL_GPUTextureFormat.D32_FLOAT_S8_UINT;
dr_depth_clear_value :: 0.0;
dr_max_shadow_verts :: 655360;
dr_near_plane :: 0.1;
dr_far_plane :: 1000.0;

dr_min_stencil_depth := 0.0;
dr_max_stencil_depth := .9999;

dr_msaa_enabled :: true;
#if dr_msaa_enabled
{
  dr_msaa_sample_count :: SDL_GPUSampleCount._4;
}
else
{
  dr_msaa_sample_count :: SDL_GPUSampleCount._1;
}

DR_Buffer :: struct
{
  next: *DR_Buffer;
  ref: *SDL_GPUBuffer;
  capacity: int;
  used: int;
}

DR_BufferRef :: struct
{
  ref: *SDL_GPUBuffer;
  offset: int;
  size: int;
}

DR_MeshInstance :: struct
{
  transform: Mat4;
  colour: V4;
  material_params: V4;
}

mesh_material_params :: (roughness: float, metallic: float, tex_coord_scale: V2) -> V4
{
  return v4(roughness, metallic, tex_coord_scale.x, tex_coord_scale.y);
}

DR_LineInstance :: struct
{
  start: V3;
  pad0: float;
  end: V3;
  pad1: float;
  colour: V4;
  props: V4;
}

DR_QuadInstance :: struct
{
  position: V3;
  pad0: float;
  colour: V4;
  props: V4;
}

line_props :: (thickness: float) -> V4
{
  result: V4;
  result.x = thickness;
  return result;
}

DR_MeshVertConstants :: struct
{
  world_to_view: Mat4;
  view_to_clip: Mat4;
  camera_pos: V3;
  pad0: float;
  viewport: V4;
  light_pos: V3;
  pad1: float;
}

DR_MeshFragConstants :: struct
{
  light: DR_LightInfo;
}

DR_MaterialDef :: struct
{
  name: string;
  col: string;
  norm: string;
  spec: string;
}

DR_MeshMaterial :: struct
{
  next: *DR_MeshMaterial;
  name: string;
  roughness := 0.5;
  metallic := 0.0;
  tex_coord_scale := V2.{1, 1};
  tex_colour: *DR_Texture;
  tex_normal: *DR_Texture;
  tex_aorm: *DR_Texture;
}

DR_DrawMesh :: struct
{
  next: *DR_DrawMesh;
  instances: DR_BufferRef;
  instances_view: [] DR_MeshInstance;
  vertices: DR_BufferRef;
  indices: DR_BufferRef;
  base_instance: int;
  base_vertex: int;
  base_index: int;
  instance_count: int;
  index_count: int;
  material: *DR_MeshMaterial;
  shadow_volume: *SH_ShadowVolume;
}

DR_DrawLines :: struct
{
  next: *DR_DrawLines;
  instances: DR_BufferRef;
  base_instance: int;
  instance_count: int;
}

DR_DrawQuads :: struct
{
  next: *DR_DrawQuads;
  instances: DR_BufferRef;
  base_instance: int;
  instance_count: int;
}

DR_Mesh :: struct
{
  next: *DR_Mesh;
  path: string;
  buffer_vertex: *DR_Buffer;
  buffer_index: *DR_Buffer;
  vertices: DR_BufferRef;
  indices: DR_BufferRef;
  vertex_count: int;
  index_count: int;
  bounds_min: V3;
  bounds_max: V3;
  positions: [] V3;
  index: [] u32;
  shadow_volume: *SH_ShadowVolume;
}

DR_Texture :: struct
{
  next: *DR_Texture;
  path: string;
  ref: *SDL_GPUTexture;
}

DR_RendererState :: struct
{
  arena: *Arena;
  device: *SDL_GPUDevice;
  window: *SDL_Window;
  resolution: V2;
  shaders: *DR_Shader;
  pipelines: *DR_Pipeline;
  buffers: *DR_Buffer;
  meshes: *DR_Mesh;
  materials: *DR_MeshMaterial;
  textures: *DR_Texture;
  mat_default: *DR_MeshMaterial;
  tex_white: *DR_Texture;
  tex_gray: *DR_Texture;
  tex_black: *DR_Texture;
  tex_normal: *DR_Texture;
  colour_target: *DR_Texture;
  resolve_target: *DR_Texture;
  depth_target: *DR_Texture;
  draw_mesh_arena: *Arena;
  draw_meshes: *DR_DrawMesh;
  draw_mesh_count: int;
  draw_lines: *DR_DrawLines;
  draw_quads: *DR_DrawQuads;
  mesh_cube: *DR_Mesh;
  mesh_line: *DR_Mesh;
  mesh_quad: *DR_Mesh;
  shader_depth: *DR_Shader;
  shader_shadow: *DR_Shader;
  shader_mesh: *DR_Shader;
  shader_lines: *DR_Shader;
  shader_quads: *DR_Shader;
  mesh_instances: *DR_Buffer;
  quad_instances: *DR_Buffer;
  line_instances: *DR_Buffer;
  pipeline_depth: *DR_Pipeline;
  pipeline_mesh: *DR_Pipeline;
  pipeline_ambient: *DR_Pipeline;
  pipeline_lines: *DR_Pipeline;
  pipeline_quads: *DR_Pipeline;
  pipeline_shadow: *DR_Pipeline;
  shadow_vertex_buffer: *DR_Buffer;
  upload_size: int;
  upload_queue: *DR_Upload;
  last_upload: *DR_Upload;
  transfer_arena: *Arena;
  transfer_buffer: *SDL_GPUTransferBuffer;
  transfer_buffer_size: int;
  sampler_set_size: [3] int;
  sampler_set: [] *SDL_GPUSampler;
  lights: [..] DR_LightInfo;
}

DR_PipelineFlags :: enum_flags
{
  None;
  Blend;
  DepthTest;
  DepthWrite;
  Indexed;
  HasWireframeVariant;
  PositionOnly;
  NoColour;
  StencilShadow;
  StencilTest;
  StencilTestInverted;
}

DR_PipelineTopology :: enum
{
  Triangles;
  Lines;
}

DR_Pipeline :: struct
{
  next: *DR_Pipeline;
  ref_base: *SDL_GPUGraphicsPipeline;
  ref_wireframe: *SDL_GPUGraphicsPipeline;
  instance_buffer: *DR_Buffer;
  flags: DR_PipelineFlags;
  topology: DR_PipelineTopology;
}

DR_Shader :: struct
{
  next: *DR_Shader;
  name: string;
  vert_meta: SDL_ShaderCross_GraphicsShaderResourceInfo;
  frag_meta: SDL_ShaderCross_GraphicsShaderResourceInfo;
  vert: *SDL_GPUShader;
  frag: *SDL_GPUShader;
}

DR_Upload :: struct
{
  next: *DR_Upload;
  size: int;
  data: *void;
  // NOTE(tt): for textures
  texture: *SDL_GPUTexture;
  width: int;
  height: int;
  x: int;
  y: int;
  generate_mipmaps: bool;
  // NOTE(tt): for buffers
  buffer: *SDL_GPUBuffer;
  offset: int;
}

DR_MeshVertex :: struct
{
  position: V3;
  tex_coord: V2;
  normal: V3;
  tangent: V4;
}

LightType :: enum u32
{
  Ambient :: 0;
  Point :: 1;
  Spot :: 2;
}

DR_LightInfo :: struct
{
  position: V3;
  type := LightType.Point;
  colour := V3.{1, 1, 1};
  brightness := 5.0;
  dir: V3;
  pad0: float;
  inner_radius := .25;
  outer_radius := 1.0;
  pad1: V2;
}

dr_renderer: *DR_RendererState;

dr_get_mesh :: (path: string) -> *DR_Mesh
{
  for each_element(dr_renderer.meshes)
  {
    if it.path == path
    {
      return it;
    }
  }
  return mesh_load(temp_arena, str_format(temp_arena, "meshes/%.mesh", path));
}

dr_get_texture :: (path: string) -> *DR_Texture
{
  for each_element(dr_renderer.textures)
  {
    if it.path == path
    {
      return it;
    }
  }
  result := tex_load(path);
  return result;
}

dr_get_material :: (name: string) -> *DR_MeshMaterial
{
  for each_element(dr_renderer.materials)
  {
    if it.name == name
    {
      return it;
    }
  }
  return dr_renderer.mat_default;
}

dr_create_mesh :: (name: string, vertices: [] DR_MeshVertex, indices: [] u32) -> *DR_Mesh
{
  capacity_vertices := size_of(DR_MeshVertex) * vertices.count;
  mesh := push_struct(dr_renderer.arena, DR_Mesh);
  stack_push(*dr_renderer.meshes, mesh);
  mesh.path = str_copy(dr_renderer.arena, name);
  mesh.buffer_vertex = dr_create_vertex_buffer(str_format(temp_arena, "%_vertices", name), capacity_vertices, null, 0, 0);
  mesh.vertices = dr_write_buffer(mesh.buffer_vertex, vertices.data, capacity_vertices);
  mesh.vertex_count = vertices.count;
  mesh.bounds_min = v3(FLOAT32_MAX);
  mesh.bounds_max = v3(-FLOAT32_MAX);
  mesh.positions = push_view(dr_renderer.arena, V3, xx vertices.count);
  for vertices
  {
    mesh.positions[it_index] = it.position;
    mesh.bounds_min = min(mesh.bounds_min, it.position);
    mesh.bounds_max = max(mesh.bounds_max, it.position);
  }
  if indices.count > 0
  {
    capacity_indices := size_of(u32) * indices.count;
    mesh.buffer_index = dr_create_index_buffer(str_format(temp_arena, "%_indices", name), capacity_indices, null, 0, 0);
    mesh.indices = dr_write_buffer(mesh.buffer_index, indices.data, capacity_indices);
    mesh.index_count = indices.count;
    mesh.index = indices;
  }
  mesh.shadow_volume = sh_generate_base_mesh(dr_renderer.arena, mesh);
  return mesh;
}

dr_create_storage_buffer :: #bake_arguments dr_create_buffer(usage = SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ);
dr_create_vertex_buffer :: #bake_arguments dr_create_buffer(usage = SDL_GPU_BUFFERUSAGE_VERTEX);
dr_create_index_buffer :: #bake_arguments dr_create_buffer(usage = SDL_GPU_BUFFERUSAGE_INDEX);
dr_create_buffer :: (name: string, capacity: int, data: *void, size: int, offset: int, usage: SDL_GPUBufferUsageFlags) -> *DR_Buffer
{
  info: SDL_GPUBufferCreateInfo;
	info.usage = usage;
	info.size = xx capacity;
	ref := SDL_CreateGPUBuffer(dr_renderer.device, *info);
	if ref
	{
	  SDL_SetGPUBufferName(dr_renderer.device, ref, cstr_from_str(temp_arena, name));
	  buffer := push_struct(dr_renderer.arena, DR_Buffer);
	  stack_push(*dr_renderer.buffers, buffer);
	  buffer.ref = ref;
	  buffer.capacity = capacity;
	  if data
	  {
	    dr_queue_buffer_upload(buffer, data, size, offset);
	    buffer.used = size;
	  }
	  return buffer;
	}
	return null;
}

dr_write_buffer :: (buffer: *DR_Buffer, data: *void, size: int) -> DR_BufferRef
{
  result: DR_BufferRef;
  if buffer.used + size <= buffer.capacity
  {
    dr_queue_buffer_upload(buffer, data, size, buffer.used);
    result.ref = buffer.ref;
    result.offset = buffer.used;
    result.size = size;
    buffer.used += size;
  }
  return result;
}

dr_reset_buffer :: (buffer: *DR_Buffer)
{
  buffer.used = 0;
}

dr_create_target_texture :: (name: string, width: int, height: int, sample_count: SDL_GPUSampleCount, format: SDL_GPUTextureFormat, usage: SDL_GPUTextureUsageFlags) -> *DR_Texture
{
  info: SDL_GPUTextureCreateInfo;
	info.type = .SDL_GPU_TEXTURETYPE_2D;
	info.format = format;
	info.width = xx width;
	info.height = xx height;
	info.layer_count_or_depth = 1;
	info.num_levels = 1;
	info.usage = usage;
	info.sample_count = sample_count;
	ref := SDL_CreateGPUTexture(dr_renderer.device, *info);
	if ref
	{
	  SDL_SetGPUTextureName(dr_renderer.device, ref, cstr_from_str(temp_arena, name));
	  texture := push_struct(dr_renderer.arena, DR_Texture);
	  texture.path = name;
	  texture.ref = ref;
	  stack_push(*dr_renderer.textures, texture);
	  return texture;
  }
	return null;
}

dr_create_draw_texture :: (name: string, width: int, height: int, format: SDL_GPUTextureFormat, data: *void) -> *DR_Texture
{
  info: SDL_GPUTextureCreateInfo;
	info.type = .SDL_GPU_TEXTURETYPE_2D;
	info.format = format;
	info.width = xx width;
	info.height = xx height;
	info.layer_count_or_depth = 1;
	info.num_levels = cast(u32)(floor(log(cast(float) max(width, height)) / log(2.0)) + 1);
	info.usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER;
	ref := SDL_CreateGPUTexture(dr_renderer.device, *info);
	if ref
	{
	  SDL_SetGPUTextureName(dr_renderer.device, ref, cstr_from_str(temp_arena, name));
	  texture := push_struct(dr_renderer.arena, DR_Texture);
	  texture.path = str_copy(dr_renderer.arena, name);
	  texture.ref = ref;
	  stack_push(*dr_renderer.textures, texture);
	  stride := 4;
	  if format ==
	  {
	    case .R8G8_UNORM;
	      stride = 2;
	  }
    dr_queue_texture_upload(ref, data, width, height, stride, info.num_levels > 1);
	  return texture;
  }
	return null;
}

dr_draw_mesh :: (mesh: *DR_Mesh, instances: [] DR_MeshInstance, material: *DR_MeshMaterial)
{
  if instances.count > 0
  {
    for * instances
    {
      it.material_params = mesh_material_params(material.roughness, material.metallic, material.tex_coord_scale);
    }
    draw_mesh := push_struct(dr_renderer.draw_mesh_arena, DR_DrawMesh);
    stack_push(*dr_renderer.draw_meshes, draw_mesh);
    dr_renderer.draw_mesh_count += instances.count;
    draw_mesh.instances = dr_write_buffer(dr_renderer.mesh_instances, instances.data, size_of(DR_MeshInstance) * instances.count);
    draw_mesh.vertices = mesh.vertices;
    draw_mesh.indices = mesh.indices;
    draw_mesh.base_instance = draw_mesh.instances.offset / size_of(DR_MeshInstance);
    draw_mesh.base_vertex = mesh.vertices.offset / size_of(DR_MeshVertex);
    draw_mesh.base_index = mesh.indices.offset / size_of(u32);
    draw_mesh.instance_count = instances.count;
    draw_mesh.index_count = mesh.indices.size / size_of(u32);
    draw_mesh.material = material;
    draw_mesh.shadow_volume = mesh.shadow_volume;
    draw_mesh.instances_view = push_copy_view(temp_arena, instances);
  }
}

dr_draw_lines :: (instances: [] DR_LineInstance)
{
  if instances.count > 0
  {
    draw_lines := push_struct(temp_arena, DR_DrawLines);
    stack_push(*dr_renderer.draw_lines, draw_lines);
    draw_lines.instances = dr_write_buffer(dr_renderer.line_instances, instances.data, size_of(DR_LineInstance) * instances.count);
    draw_lines.base_instance = draw_lines.instances.offset / size_of(DR_LineInstance);
    draw_lines.instance_count = instances.count;
  }
}

dr_draw_quads :: (instances: [] DR_QuadInstance)
{
  if instances.count > 0
  {
    draw_quads := push_struct(temp_arena, DR_DrawQuads);
    stack_push(*dr_renderer.draw_quads, draw_quads);
    draw_quads.instances = dr_write_buffer(dr_renderer.quad_instances, instances.data, size_of(DR_QuadInstance) * instances.count);
    draw_quads.base_instance = draw_quads.instances.offset / size_of(DR_QuadInstance);
    draw_quads.instance_count = instances.count;
  }
}

quad_props :: (size: V2, border: float) -> V4
{
  result: V4;
  result.xy = size;
  result.z = border;
  return result;
}

dr_draw_bounds :: (bounds_min: V3, bounds_max: V3, colour: V4, thickness: float)
{
  line_instances := DR_LineInstance.[
    .{start=v3(bounds_min.x, bounds_min.y, bounds_min.z), end=v3(bounds_max.x, bounds_min.y, bounds_min.z), colour=colour, props=line_props(thickness)},
    .{start=v3(bounds_max.x, bounds_min.y, bounds_min.z), end=v3(bounds_max.x, bounds_min.y, bounds_max.z), colour=colour, props=line_props(thickness)},
    .{start=v3(bounds_max.x, bounds_min.y, bounds_max.z), end=v3(bounds_min.x, bounds_min.y, bounds_max.z), colour=colour, props=line_props(thickness)},
    .{start=v3(bounds_min.x, bounds_min.y, bounds_max.z), end=v3(bounds_min.x, bounds_min.y, bounds_min.z), colour=colour, props=line_props(thickness)},
    .{start=v3(bounds_min.x, bounds_max.y, bounds_min.z), end=v3(bounds_max.x, bounds_max.y, bounds_min.z), colour=colour, props=line_props(thickness)},
    .{start=v3(bounds_max.x, bounds_max.y, bounds_min.z), end=v3(bounds_max.x, bounds_max.y, bounds_max.z), colour=colour, props=line_props(thickness)},
    .{start=v3(bounds_max.x, bounds_max.y, bounds_max.z), end=v3(bounds_min.x, bounds_max.y, bounds_max.z), colour=colour, props=line_props(thickness)},
    .{start=v3(bounds_min.x, bounds_max.y, bounds_max.z), end=v3(bounds_min.x, bounds_max.y, bounds_min.z), colour=colour, props=line_props(thickness)},
    .{start=v3(bounds_min.x, bounds_min.y, bounds_min.z), end=v3(bounds_min.x, bounds_max.y, bounds_min.z), colour=colour, props=line_props(thickness)},
    .{start=v3(bounds_max.x, bounds_min.y, bounds_min.z), end=v3(bounds_max.x, bounds_max.y, bounds_min.z), colour=colour, props=line_props(thickness)},
    .{start=v3(bounds_max.x, bounds_min.y, bounds_max.z), end=v3(bounds_max.x, bounds_max.y, bounds_max.z), colour=colour, props=line_props(thickness)},
    .{start=v3(bounds_min.x, bounds_min.y, bounds_max.z), end=v3(bounds_min.x, bounds_max.y, bounds_max.z), colour=colour, props=line_props(thickness)},
  ];
  dr_draw_lines(line_instances);
}

dr_draw_sphere :: (center: V3, radius: float, segments: int, colour: V4, thickness: float)
{
  dr_draw_circle(center, v3(1, 0, 0), radius, segments, colour, thickness);
  dr_draw_circle(center, v3(0, 1, 0), radius, segments, colour, thickness);
  dr_draw_circle(center, v3(0, 0, 1), radius, segments, colour, thickness);
}

dr_draw_capsule :: (center: V3, radius: float, height: float, colour: V4, thickness: float)
{
  top := center + v3(0, 0, height * 0.5);
  bottom := center - v3(0, 0, height * 0.5);
  dr_draw_circle(top, v3(1, 0, 0), radius, 8, colour, thickness);
  dr_draw_circle(top, v3(0, 1, 0), radius, 8, colour, thickness);
  dr_draw_circle(bottom, v3(1, 0, 0), radius, 8, colour, thickness);
  dr_draw_circle(bottom, v3(0, 1, 0), radius, 8, colour, thickness);
  line_instances := DR_LineInstance.[
    .{start=top + v3(radius, 0, 0), end=bottom + v3(radius, 0, 0), colour=colour, props=line_props(thickness)},
    .{start=top - v3(radius, 0, 0), end=bottom - v3(radius, 0, 0), colour=colour, props=line_props(thickness)},
    .{start=top + v3(0, radius, 0), end=bottom + v3(0, radius, 0), colour=colour, props=line_props(thickness)},
    .{start=top - v3(0, radius, 0), end=bottom - v3(0, radius, 0), colour=colour, props=line_props(thickness)},
  ];
  dr_draw_lines(line_instances);
}

dr_draw_circle :: (center: V3, normal: V3, radius: float, segments: int, colour: V4, thickness: float)
{
  n := normalize(normal);
  tmp := ifx abs(n.x) < 0.99 then v3(1,0,0) else v3(0,1,0);
  tangent := normalize(cross(n, tmp));
  bitangent := cross(n, tangent);
  line_instances := push_view(temp_arena, DR_LineInstance, xx segments);
  step := (2.0 * PI) / cast(float) segments;
  prev := center + tangent*radius;
  for 1..segments
  {
    angle := step * cast(float) it;
    p := center + (tangent * cos(angle) + bitangent * sin(angle)) * radius;
    line_instances[it-1] = .{
      start = prev,
      end = p,
      colour = colour,
      props = line_props(thickness),
    };
    prev = p;
  }
  dr_draw_lines(line_instances);
}

dr_create_pipeline :: (shader: *DR_Shader, instance_buffer: *DR_Buffer, flags: DR_PipelineFlags, topology: DR_PipelineTopology) -> *DR_Pipeline
{
  colour_target_desc: SDL_GPUColorTargetDescription;
  colour_target_desc.format = dr_colour_format;
  if flags & .Blend
  {
    colour_target_desc.blend_state.enable_blend = true;
    colour_target_desc.blend_state.src_color_blendfactor = .ONE;
    colour_target_desc.blend_state.dst_color_blendfactor = .ONE;
    colour_target_desc.blend_state.color_blend_op = .ADD;
    colour_target_desc.blend_state.src_alpha_blendfactor = .ONE;
    colour_target_desc.blend_state.dst_alpha_blendfactor = .ONE;
    colour_target_desc.blend_state.alpha_blend_op = .ADD;
  }
  VertAttributeDesc :: struct
  {
    offset: u32;
    format: SDL_GPUVertexElementFormat;
  };
  vert_attribute_desc :: VertAttributeDesc.[
    .{#run offset_of(DR_MeshVertex, "position"),  .FLOAT3},
    .{#run offset_of(DR_MeshVertex, "tex_coord"), .FLOAT2},
    .{#run offset_of(DR_MeshVertex, "normal"),    .FLOAT3},
    .{#run offset_of(DR_MeshVertex, "tangent"),   .FLOAT4},
  ];
  attributes: [vert_attribute_desc.count] SDL_GPUVertexAttribute;
  if flags & .PositionOnly
  {
    attributes[0].location = 0;
    attributes[0].buffer_slot = 0;
    attributes[0].offset = vert_attribute_desc[0].offset;
    attributes[0].format = vert_attribute_desc[0].format;
  }
  else
  {
    for * vert_attribute_desc
    {
      attributes[it_index].location = xx it_index;
      attributes[it_index].buffer_slot = 0;
      attributes[it_index].offset = it.offset;
      attributes[it_index].format = it.format;
    }
  }
  vertex_buffer_desc: SDL_GPUVertexBufferDescription;
  vertex_buffer_desc.slot = 0;
  if flags & .StencilShadow
  {
    vertex_buffer_desc.pitch = size_of(V3);
  }
  else
  {
    vertex_buffer_desc.pitch = size_of(DR_MeshVertex);
  }
  vertex_buffer_desc.input_rate = .VERTEX;
  info_base: SDL_GPUGraphicsPipelineCreateInfo;
  info_base.vertex_input_state.vertex_buffer_descriptions = *vertex_buffer_desc;
  info_base.vertex_input_state.num_vertex_buffers = 1;
  info_base.vertex_input_state.vertex_attributes = attributes.data;
  info_base.vertex_input_state.num_vertex_attributes = xx ifx flags & .PositionOnly then 1 else vert_attribute_desc.count;
  if !cast(bool) flags & .NoColour
  {
    info_base.target_info.color_target_descriptions = *colour_target_desc;
    info_base.target_info.num_color_targets = 1;
  }
  info_base.rasterizer_state.cull_mode = .BACK;
  info_base.rasterizer_state.fill_mode = .FILL;
  info_base.rasterizer_state.enable_depth_clip = true;
  info_base.multisample_state.sample_count = dr_msaa_sample_count;
  if topology ==
  {
    case .Triangles;
      info_base.primitive_type = .TRIANGLELIST;
  }
  info_base.vertex_shader = shader.vert;
  info_base.fragment_shader = shader.frag;
  if (flags & .DepthTest) || (flags & .DepthWrite)
  {
    info_base.target_info.depth_stencil_format = dr_depth_format;
    info_base.target_info.has_depth_stencil_target = true;
  }
  info_base.depth_stencil_state.enable_depth_test = cast(bool) (flags & .DepthTest);
  info_base.depth_stencil_state.enable_depth_write = cast(bool) (flags & .DepthWrite);
  if flags & .DepthWrite
  {
    info_base.depth_stencil_state.compare_op = .GREATER;
  }
  else
  {
    info_base.depth_stencil_state.compare_op = .GREATER_OR_EQUAL;
  }
  if flags & .StencilShadow
  {
    info_base.rasterizer_state.cull_mode = .NONE;
    info_base.depth_stencil_state.enable_stencil_test = true;
    info_base.depth_stencil_state.compare_mask = 0xff;
    info_base.depth_stencil_state.write_mask = 0xff;
    info_base.depth_stencil_state.back_stencil_state.compare_op = .ALWAYS;
    info_base.depth_stencil_state.back_stencil_state.fail_op = .KEEP;
    info_base.depth_stencil_state.back_stencil_state.pass_op = .KEEP;
    info_base.depth_stencil_state.back_stencil_state.depth_fail_op = .INCREMENT_AND_WRAP;
    info_base.depth_stencil_state.front_stencil_state.compare_op = .ALWAYS;
    info_base.depth_stencil_state.front_stencil_state.fail_op = .KEEP;
    info_base.depth_stencil_state.front_stencil_state.pass_op = .KEEP;
    info_base.depth_stencil_state.front_stencil_state.depth_fail_op = .DECREMENT_AND_WRAP;
  }
  if flags & .StencilTest
  {
    info_base.depth_stencil_state.enable_stencil_test = true;
    info_base.depth_stencil_state.compare_mask = 0xff;
    info_base.depth_stencil_state.back_stencil_state.compare_op = .EQUAL;
    info_base.depth_stencil_state.front_stencil_state.compare_op = .EQUAL;
    info_base.depth_stencil_state.back_stencil_state.fail_op = .KEEP;
    info_base.depth_stencil_state.back_stencil_state.pass_op = .KEEP;
    info_base.depth_stencil_state.back_stencil_state.depth_fail_op = .KEEP;
    info_base.depth_stencil_state.front_stencil_state.fail_op = .KEEP;
    info_base.depth_stencil_state.front_stencil_state.pass_op = .KEEP;
    info_base.depth_stencil_state.front_stencil_state.depth_fail_op = .KEEP;
  }
  if flags & .StencilTestInverted
  {
    info_base.depth_stencil_state.enable_stencil_test = true;
    info_base.depth_stencil_state.compare_mask = 0xff;
    info_base.depth_stencil_state.back_stencil_state.compare_op = .NOT_EQUAL;
    info_base.depth_stencil_state.front_stencil_state.compare_op = .NOT_EQUAL;
    info_base.depth_stencil_state.back_stencil_state.fail_op = .KEEP;
    info_base.depth_stencil_state.back_stencil_state.pass_op = .KEEP;
    info_base.depth_stencil_state.back_stencil_state.depth_fail_op = .KEEP;
    info_base.depth_stencil_state.front_stencil_state.fail_op = .KEEP;
    info_base.depth_stencil_state.front_stencil_state.pass_op = .KEEP;
    info_base.depth_stencil_state.front_stencil_state.depth_fail_op = .KEEP;
  }
  ref_base := SDL_CreateGPUGraphicsPipeline(dr_renderer.device, *info_base);
  pipeline := push_struct(dr_renderer.arena, DR_Pipeline);
  stack_push(*dr_renderer.pipelines, pipeline);
  pipeline.instance_buffer = instance_buffer;
  pipeline.flags = flags;
  pipeline.topology = topology;
  pipeline.ref_base = ref_base;
  if flags & .HasWireframeVariant
  {
    info_wireframe := info_base;
    info_wireframe.rasterizer_state.fill_mode = .LINE;
    ref_wireframe := SDL_CreateGPUGraphicsPipeline(dr_renderer.device, *info_wireframe);
    pipeline.ref_wireframe = ref_wireframe;
  }
  return pipeline;
}

dr_create_builtin_meshes :: () {
  cube_vertices := DR_MeshVertex.[
    .{position=v3(-0.5,-0.5, 0.5), tex_coord=v2(0,0), normal=v3(0,0,1),  tangent=v4( 1, 0, 0, 1)},
    .{position=v3( 0.5,-0.5, 0.5), tex_coord=v2(1,0), normal=v3(0,0,1),  tangent=v4( 1, 0, 0, 1)},
    .{position=v3( 0.5, 0.5, 0.5), tex_coord=v2(1,1), normal=v3(0,0,1),  tangent=v4( 1, 0, 0, 1)},
    .{position=v3(-0.5, 0.5, 0.5), tex_coord=v2(0,1), normal=v3(0,0,1),  tangent=v4( 1, 0, 0, 1)},
    .{position=v3(-0.5,-0.5,-0.5), tex_coord=v2(0,1), normal=v3(0,0,-1), tangent=v4( 1, 0, 0, 1)},
    .{position=v3(-0.5, 0.5,-0.5), tex_coord=v2(0,0), normal=v3(0,0,-1), tangent=v4( 1, 0, 0, 1)},
    .{position=v3( 0.5, 0.5,-0.5), tex_coord=v2(1,0), normal=v3(0,0,-1), tangent=v4( 1, 0, 0, 1)},
    .{position=v3( 0.5,-0.5,-0.5), tex_coord=v2(1,1), normal=v3(0,0,-1), tangent=v4( 1, 0, 0, 1)},
    .{position=v3( 0.5,-0.5,-0.5), tex_coord=v2(0,1), normal=v3(1,0,0),  tangent=v4( 0,-1, 0, 1)},
    .{position=v3( 0.5, 0.5,-0.5), tex_coord=v2(1,1), normal=v3(1,0,0),  tangent=v4( 0,-1, 0, 1)},
    .{position=v3( 0.5, 0.5, 0.5), tex_coord=v2(1,0), normal=v3(1,0,0),  tangent=v4( 0,-1, 0, 1)},
    .{position=v3( 0.5,-0.5, 0.5), tex_coord=v2(0,0), normal=v3(1,0,0),  tangent=v4( 0,-1, 0, 1)},
    .{position=v3(-0.5,-0.5,-0.5), tex_coord=v2(1,1), normal=v3(-1,0,0), tangent=v4( 0, 1, 0, 1)},
    .{position=v3(-0.5,-0.5, 0.5), tex_coord=v2(1,0), normal=v3(-1,0,0), tangent=v4( 0, 1, 0, 1)},
    .{position=v3(-0.5, 0.5, 0.5), tex_coord=v2(0,0), normal=v3(-1,0,0), tangent=v4( 0, 1, 0, 1)},
    .{position=v3(-0.5, 0.5,-0.5), tex_coord=v2(0,1), normal=v3(-1,0,0), tangent=v4( 0, 1, 0, 1)},
    .{position=v3(-0.5, 0.5,-0.5), tex_coord=v2(1,1), normal=v3(0,1,0),  tangent=v4( 1, 0, 0, 1)},
    .{position=v3(-0.5, 0.5, 0.5), tex_coord=v2(1,0), normal=v3(0,1,0),  tangent=v4( 1, 0, 0, 1)},
    .{position=v3( 0.5, 0.5, 0.5), tex_coord=v2(0,0), normal=v3(0,1,0),  tangent=v4( 1, 0, 0, 1)},
    .{position=v3( 0.5, 0.5,-0.5), tex_coord=v2(0,1), normal=v3(0,1,0),  tangent=v4( 1, 0, 0, 1)},
    .{position=v3(-0.5,-0.5,-0.5), tex_coord=v2(0,1), normal=v3(0,-1,0), tangent=v4(-1, 0, 0, 1)},
    .{position=v3( 0.5,-0.5,-0.5), tex_coord=v2(1,1), normal=v3(0,-1,0), tangent=v4(-1, 0, 0, 1)},
    .{position=v3( 0.5,-0.5, 0.5), tex_coord=v2(1,0), normal=v3(0,-1,0), tangent=v4(-1, 0, 0, 1)},
    .{position=v3(-0.5,-0.5, 0.5), tex_coord=v2(0,0), normal=v3(0,-1,0), tangent=v4(-1, 0, 0, 1)},
  ];
  cube_indices := u32.[
    0,1,2, 0,2,3,
    4,5,6, 4,6,7,
    8,9,10, 8,10,11,
    12,13,14, 12,14,15,
    16,17,18, 16,18,19,
    20,21,22, 20,22,23,
  ];
  dr_renderer.mesh_cube = dr_create_mesh("dr_default_cube_mesh", cube_vertices, cube_indices);
  line_vertices := DR_MeshVertex.[
    .{position = v3(-1,  2, 0), tex_coord=v2(-1,  2)},
    .{position = v3( 1,  2, 0), tex_coord=v2( 1,  2)},
    .{position = v3(-1,  1, 0), tex_coord=v2(-1,  1)},
    .{position = v3( 1,  1, 0), tex_coord=v2( 1,  1)},
    .{position = v3(-1,  1, 0), tex_coord=v2(-1,  1)},
    .{position = v3( 1,  1, 0), tex_coord=v2( 1,  1)},
    .{position = v3(-1,  0, 0), tex_coord=v2(-1,  0)},
    .{position = v3( 1,  0, 0), tex_coord=v2( 1,  0)},
    .{position = v3(-1,  0, 0), tex_coord=v2(-1,  0)},
    .{position = v3( 1,  0, 0), tex_coord=v2( 1,  0)},
    .{position = v3(-1, -1, 0), tex_coord=v2(-1, -1)},
    .{position = v3( 1, -1, 0), tex_coord=v2( 1, -1)},
  ];
  line_indices := u32.[
    0, 2, 1, 2, 3, 1,
    4, 6, 5, 6, 7, 5,
    8, 10, 9, 10, 11, 9,
  ];
  dr_renderer.mesh_line = dr_create_mesh("dr_line", line_vertices, line_indices);
  quad_vertices := DR_MeshVertex.[
    .{position = v3(-.5, -.5, 0), tex_coord=v2(0, 0)},
    .{position = v3( .5, -.5, 0), tex_coord=v2(1, 0)},
    .{position = v3( .5,  .5, 0), tex_coord=v2(1, 1)},
    .{position = v3(-.5,  .5, 0), tex_coord=v2(0, 1)},
  ];
  quad_indices := u32.[
    0, 1, 2, 2, 3, 0
  ];
  dr_renderer.mesh_quad = dr_create_mesh("dr_quad", quad_vertices, quad_indices);
}

dr_create_builtin_textures :: ()
{
  dr_renderer.tex_white = dr_create_draw_texture("dr_default_white_texture", 1, 1, .R8G8B8A8_UNORM, u8.[255, 255, 255, 255].data);
  dr_renderer.tex_gray = dr_create_draw_texture("dr_default_gray_texture", 1, 1, .R8G8B8A8_UNORM, u8.[128, 128, 128, 255].data);
  dr_renderer.tex_black = dr_create_draw_texture("dr_default_black_texture", 1, 1, .R8G8B8A8_UNORM, u8.[0, 0, 0, 255].data);
  dr_renderer.tex_normal = dr_create_draw_texture("dr_default_normal_texture", 1, 1, .R8G8B8A8_UNORM, u8.[0, 128, 0, 0].data);
}

dr_create_sampler_set :: ()
{
  filters := SDL_GPUFilter.[
    .NEAREST,
    .LINEAR
  ];
  mip_filters := SDL_GPUSamplerMipmapMode.[
    .NEAREST,
    .LINEAR
  ];
  address_modes := SDL_GPUSamplerAddressMode.[
    .REPEAT,
    .MIRRORED_REPEAT,
    .CLAMP_TO_EDGE
  ];
  dr_renderer.sampler_set_size = .[filters.count, mip_filters.count, address_modes.count];
  dr_renderer.sampler_set = push_view(dr_renderer.arena, *SDL_GPUSampler, filters.count * mip_filters.count * address_modes.count);
  for filter: filters
  {
    for mip_filter: mip_filters
    {
      for address_mode: address_modes
      {
        info: SDL_GPUSamplerCreateInfo;
        info.min_filter = filter;
        info.mag_filter = filter;
        info.mipmap_mode = mip_filter;
        info.address_mode_u = address_mode;
        info.address_mode_v = address_mode;
        info.max_lod = 99;
        sampler := SDL_CreateGPUSampler(dr_renderer.device, *info);
        index := cast(int) filter +
                 cast(int) mip_filter*filters.count +
                 cast(int) address_mode*mip_filters.count;
        dr_renderer.sampler_set[index] = sampler;
      }
    }
  }
}

dr_add_light :: (info: DR_LightInfo)
{
  array_add(*dr_renderer.lights, info);
}

dr_init :: (window: *SDL_Window, device: *SDL_GPUDevice, resolution: V2)
{
  arena := create_arena();
  dr_renderer = push_struct(arena, DR_RendererState);
  dr_renderer.arena = arena;
  dr_renderer.window = window;
  dr_renderer.device = device;
  dr_renderer.resolution = resolution;
  dr_renderer.transfer_arena = create_arena();
  dr_renderer.draw_mesh_arena = create_arena();
  dr_renderer.shader_depth = dr_get_shader("depth");
  dr_renderer.shader_mesh = dr_get_shader("mesh");
  dr_renderer.shader_lines = dr_get_shader("lines");
  dr_renderer.shader_quads = dr_get_shader("quads");
  dr_renderer.shader_shadow = dr_get_shader("shadow");
  dr_renderer.mesh_instances = dr_create_storage_buffer("mesh_instances", size_of(DR_MeshInstance) * dr_max_mesh_instances, null, 0, 0);
  dr_renderer.line_instances = dr_create_storage_buffer("line_instances", size_of(DR_LineInstance) * dr_max_line_instances, null, 0, 0);
  dr_renderer.quad_instances = dr_create_storage_buffer("quad_instances", size_of(DR_QuadInstance) * dr_max_quad_instances, null, 0, 0);
  dr_renderer.pipeline_depth = dr_create_pipeline(dr_renderer.shader_depth, dr_renderer.mesh_instances, .DepthTest | .DepthWrite | .PositionOnly | .NoColour, .Triangles);
  dr_renderer.pipeline_shadow = dr_create_pipeline(dr_renderer.shader_shadow, null, .DepthTest | .StencilShadow | .PositionOnly | .NoColour, .Triangles);
  dr_renderer.pipeline_mesh = dr_create_pipeline(dr_renderer.shader_mesh, dr_renderer.mesh_instances, .Blend | .DepthTest | .StencilTest, .Triangles);
  dr_renderer.pipeline_ambient = dr_create_pipeline(dr_renderer.shader_mesh, dr_renderer.mesh_instances, .Blend | .DepthTest | .StencilTestInverted, .Triangles);
  dr_renderer.pipeline_lines = dr_create_pipeline(dr_renderer.shader_lines, dr_renderer.line_instances, .DepthTest, .Triangles);
  dr_renderer.pipeline_quads = dr_create_pipeline(dr_renderer.shader_quads, dr_renderer.quad_instances, .None, .Triangles);
  dr_renderer.shadow_vertex_buffer =  dr_create_vertex_buffer("shadow_vertices", size_of(V3) * dr_max_shadow_verts, null, 0, 0);
  if dr_msaa_enabled
  {
    dr_renderer.colour_target = dr_create_target_texture("colour_target", xx resolution.x, xx resolution.y, dr_msaa_sample_count, dr_colour_format, SDL_GPU_TEXTUREUSAGE_COLOR_TARGET);
    dr_renderer.resolve_target = dr_create_target_texture("resolve_target", xx resolution.x, xx resolution.y, ._1, dr_colour_format, SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER);
  }
  dr_renderer.depth_target = dr_create_target_texture("depth_target", xx resolution.x, xx resolution.y, dr_msaa_sample_count, dr_depth_format, SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET);
  dr_renderer.lights.allocator = arena_allocator(temp_arena);
  dr_create_builtin_meshes();
  dr_create_builtin_textures();
  dr_create_sampler_set();
  dr_renderer.mat_default = dr_create_material();
  dr_renderer.mat_default.name = "dr_default_material";
  read_materials();
}

dr_quit :: ()
{
  write_materials();
}

dr_create_material :: () -> *DR_MeshMaterial
{
  mat := push_struct(dr_renderer.arena, DR_MeshMaterial);
  stack_push(*dr_renderer.materials, mat);
  mat.tex_colour = dr_renderer.tex_white;
  mat.tex_normal = dr_renderer.tex_normal;
  mat.tex_aorm = dr_renderer.tex_white;
  return mat;
}

read_materials :: ()
{
  texture_files := file_list("textures");
  for texture_files
  {
    if !contains(it, "_norm") && !contains(it, "_aorm")
    {
      path, base, ext := path_decomp(it);
      col := str_temp_format("textures/%.%", base, ext);
      norm := str_temp_format("textures/%_norm.%", base, ext);
      aorm := str_temp_format("textures/%_aorm.%", base, ext);
      if !file_exists(str_temp_format("%", norm))
      {
        norm = "dr_default_normal_texture";
      }
      if !file_exists(str_temp_format("%", aorm))
      {
        norm = "dr_default_white_texture";
      }
      if !file_exists(str_temp_format("materials/%.txt", base))
      {
        builder: String_Builder;
        print_to_builder(*builder, "name: %\n", base);
        print_to_builder(*builder, "tex_colour: %\n", col);
        print_to_builder(*builder, "tex_normal: %\n", norm);
        print_to_builder(*builder, "tex_aorm: %\n", aorm);
        mat := builder_to_string(*builder);
        write_entire_file(str_temp_format("materials/%.txt", base), mat);
      }
    }
  }
  for file_list("materials")
  {
    file_data, ok := read_entire_file(it,, arena_allocator(temp_arena));
    if ok
    {
      mat := push_struct(dr_renderer.arena, DR_MeshMaterial);
      stack_push(*dr_renderer.materials, mat);
      _, lines := file_to_array("", true, true, false, true, file_data);
      for lines
      {
        key, right := break_by_spaces(it);
        key.count -= 1;
        if key ==
        {
          case "name";
            mat.name = str_copy(dr_renderer.arena, right);
          case "roughness";
            mat.roughness = parse_float(*right);
          case "metallic";
            mat.metallic = parse_float(*right);
          case "tex_coord_scale";
            mat.tex_coord_scale = parse_v2(right);
          case "tex_colour";
            mat.tex_colour = dr_get_texture(right);
          case "tex_normal";
            mat.tex_normal = dr_get_texture(right);
          case "tex_aorm";
            mat.tex_aorm = dr_get_texture(right);
        }
      }
    }
  }
}

write_materials :: ()
{
  for each_element(dr_renderer.materials)
  {
    if it.name != "dr_default_material"
    {
      path := str_format(temp_arena, "materials/%.txt", it.name);
      file := file_open(path, for_writing = true);
      defer file_close(*file);
      file_print_line(*file, "name: %", it.name);
      file_print_line(*file, "roughness: %", it.roughness);
      file_print_line(*file, "metallic: %", it.metallic);
      file_print(*file, "tex_coord_scale", it.tex_coord_scale);
      if it.tex_colour file_print_line(*file, "tex_colour: %", it.tex_colour.path);
      if it.tex_normal file_print_line(*file, "tex_normal: %", it.tex_normal.path);
      if it.tex_aorm file_print_line(*file, "tex_aorm: %", it.tex_aorm.path);
    }
  }
}

dr_get_sampler :: (filter: SDL_GPUFilter, mip_filter: SDL_GPUSamplerMipmapMode, address_mode: SDL_GPUSamplerAddressMode) -> *SDL_GPUSampler
{
  index := cast(int) filter +
    cast(int) mip_filter*dr_renderer.sampler_set_size[0] +
    cast(int) address_mode*dr_renderer.sampler_set_size[1];
  return dr_renderer.sampler_set[index];
}

dr_queue_buffer_upload :: (dest: *DR_Buffer, data: *void, size: int, offset: int)
{
  upload := push_struct(dr_renderer.transfer_arena, DR_Upload);
  queue_push(*dr_renderer.upload_queue, *dr_renderer.last_upload, upload);
  upload.size = size;
  upload.data = push_copy(dr_renderer.transfer_arena, data, xx size);
  upload.offset = offset;
  upload.buffer = dest.ref;
  dr_renderer.upload_size += size;
}

dr_queue_texture_upload :: (dest: *SDL_GPUTexture, data: *void, width: int, height: int, stride: int, generate_mipmaps: bool, x := 0, y := 0)
{
  upload := push_struct(dr_renderer.transfer_arena, DR_Upload);
  queue_push(*dr_renderer.upload_queue, *dr_renderer.last_upload, upload);
  upload.size = width * height * stride;
  upload.width = width;
  upload.height = height;
  upload.data = push_copy(dr_renderer.transfer_arena, data, xx upload.size);
  upload.x = x;
  upload.y = y;
  upload.texture = dest;
  upload.generate_mipmaps = generate_mipmaps;
  dr_renderer.upload_size += upload.size;
}

dr_process_upload_queue :: (command_buffer: *SDL_GPUCommandBuffer)
{
  // NOTE(tt): resize transfer buffer if needed
  if (dr_renderer.transfer_buffer_size < dr_renderer.upload_size)
  {
    if (dr_renderer.transfer_buffer)
    {
      SDL_ReleaseGPUTransferBuffer(dr_renderer.device, dr_renderer.transfer_buffer);
    }
    dr_renderer.transfer_buffer = dr_create_transfer_buffer(dr_renderer.upload_size);
    dr_renderer.transfer_buffer_size = dr_renderer.upload_size;
  }
  // NOTE(tt): copy data to transfer buffer
  mapping := SDL_MapGPUTransferBuffer(dr_renderer.device, dr_renderer.transfer_buffer, true);
  cursor := 0;
  for each_element(dr_renderer.upload_queue)
  {
    copy(mapping + cursor, it.data, xx it.size);
    cursor += it.size;
  }
  SDL_UnmapGPUTransferBuffer(dr_renderer.device, dr_renderer.transfer_buffer);
  // NOTE(tt): copy from transfer buffer to GPU resources
  cursor = 0;
  copy_pass := SDL_BeginGPUCopyPass(command_buffer);
  for each_element(dr_renderer.upload_queue)
  {
    if it.buffer
    {
      location: SDL_GPUTransferBufferLocation;
      location.transfer_buffer = dr_renderer.transfer_buffer;
      location.offset = xx cursor;
      region: SDL_GPUBufferRegion;
      region.buffer = it.buffer;
      region.size = xx it.size;
      region.offset = xx it.offset;
      SDL_UploadToGPUBuffer(copy_pass, *location, *region, false);
    }
    else if it.texture
    {
      location: SDL_GPUTextureTransferInfo;
      location.transfer_buffer = dr_renderer.transfer_buffer;
      location.offset = xx cursor;
      location.pixels_per_row = xx it.width;
      location.rows_per_layer = xx it.height;
      region: SDL_GPUTextureRegion;
      region.texture = it.texture;
      region.w = xx it.width;
      region.h = xx it.height;
      region.x = xx it.x;
      region.y = xx it.y;
      region.d = 1;
      SDL_UploadToGPUTexture(copy_pass, *location, *region, false);
    }
    cursor += it.size;
  }
  SDL_EndGPUCopyPass(copy_pass);
  for each_element(dr_renderer.upload_queue)
  {
    if it.texture && it.generate_mipmaps
    {
      SDL_GenerateMipmapsForGPUTexture(command_buffer, it.texture);
    }
  }
  dr_renderer.upload_size = 0;
  queue_reset(*dr_renderer.upload_queue, *dr_renderer.last_upload);
  arena_reset(dr_renderer.transfer_arena);
}

dr_run_clear_pass :: (swap_texture: *SDL_GPUTexture, command_buffer: *SDL_GPUCommandBuffer)
{
  colour_target := SDL_GPUColorTargetInfo.{
    texture = swap_texture,
    clear_color = .{0, 0, 0, 0},
    load_op = .CLEAR,
    store_op = .STORE,
    cycle = true
  };
  depth_target := SDL_GPUDepthStencilTargetInfo.{
    texture = dr_renderer.depth_target.ref,
    clear_depth = dr_depth_clear_value,
    load_op = .CLEAR,
    store_op = .STORE,
    stencil_load_op = .DONT_CARE,
    stencil_store_op = .DONT_CARE,
    cycle = true
  };
  pass := SDL_BeginGPURenderPass(command_buffer, *colour_target, 1, *depth_target);
  SDL_EndGPURenderPass(pass);
}

dr_run_depth_prepass :: (command_buffer: *SDL_GPUCommandBuffer, vert_constants: DR_MeshVertConstants)
{
  depth_target := SDL_GPUDepthStencilTargetInfo.{
    texture = dr_renderer.depth_target.ref,
    clear_depth = dr_depth_clear_value,
    load_op = .LOAD,
    store_op = .STORE
  };
  pass := SDL_BeginGPURenderPass(command_buffer, null, 0, *depth_target);
  SDL_BindGPUGraphicsPipeline(pass, dr_renderer.pipeline_depth.ref_base);
  SDL_PushGPUVertexUniformData(command_buffer, 0, *vert_constants, xx size_of(DR_MeshVertConstants));
  SDL_BindGPUVertexStorageBuffers(pass, 0, *dr_renderer.mesh_instances.ref, 1);
  for each_element(dr_renderer.draw_meshes)
  {
    binding_vertex := SDL_GPUBufferBinding.{it.vertices.ref, 0};
    binding_index := SDL_GPUBufferBinding.{it.indices.ref, 0};
    SDL_BindGPUVertexBuffers(pass, 0, *binding_vertex, 1);
    SDL_BindGPUIndexBuffer(pass, *binding_index, ._32BIT);
    SDL_DrawGPUIndexedPrimitives(pass,
      xx it.index_count,
      xx it.instance_count,
      xx it.base_index,
      xx it.base_vertex,
      xx it.base_instance
    );
  }
  SDL_EndGPURenderPass(pass);
}

ShadowVolume :: struct
{
  base_vert: int;
  vert_count: int;
  instance_count: int;
  base_instance: int;
}

DrawCall :: struct
{
  light: DR_LightInfo;
  volumes: [..] ShadowVolume;
}

dr_draw_frame :: (camera: *CAM_Base)
{
  ImGui.Begin("Renderer");
  ImGui.DragFloat("Stencil min depth", *dr_min_stencil_depth, v_min=0.0, v_max=1.0, v_speed=.00001);
  ImGui.DragFloat("Stencil max depth", *dr_max_stencil_depth, v_min=0.0, v_max=1.0, v_speed=.00001);
  ImGui.End();

  command_buffer := SDL_AcquireGPUCommandBuffer(dr_renderer.device);
  swap_texture: *SDL_GPUTexture;
  swap_width, swap_height: u32;
  SDL_WaitAndAcquireGPUSwapchainTexture(command_buffer, dr_renderer.window, *swap_texture, *swap_width, *swap_height);
  if (swap_texture)
  {
    draw_calls := push_view(temp_arena, DrawCall, xx dr_renderer.lights.count);
    for * draw_calls
    {
      it.volumes.allocator = arena_allocator(temp_arena);
    }
    position_count := 0;
    for light, light_index: dr_renderer.lights
    {
      for each_element(dr_renderer.draw_meshes)
      {
        for inst: it.instances_view
        {
          positions := sh_build_shadow_mesh(temp_arena, it.shadow_volume, light, inst.transform);
          dr_queue_buffer_upload(dr_renderer.shadow_vertex_buffer, positions.data, size_of(V3) * positions.count, size_of(V3) * position_count);
          volume: ShadowVolume;
          volume.base_vert = position_count;
          volume.vert_count = positions.count;
          volume.instance_count = it.instance_count;
          volume.base_instance = it.base_instance;
          position_count += positions.count;
          draw_calls[light_index].light = light;
          array_add(*draw_calls[light_index].volumes, volume);
        }
      }
    }
    dr_process_upload_queue(command_buffer);
    vert_constants: DR_MeshVertConstants;
    vert_constants.world_to_view = camera.view;
    vert_constants.view_to_clip = camera.projection;
    vert_constants.camera_pos = camera.position;
    vert_constants.viewport.xy = dr_renderer.resolution;
    vert_constants.viewport.zw = v2(dr_near_plane, dr_far_plane);
    colour_target_texture := ifx dr_msaa_enabled then dr_renderer.colour_target.ref else swap_texture;
    dr_run_clear_pass(colour_target_texture, command_buffer);
    dr_run_depth_prepass(command_buffer, vert_constants);
    for light, light_index: dr_renderer.lights
    {
      shadow_volumes := draw_calls[light_index].volumes;
      depth_target := SDL_GPUDepthStencilTargetInfo.{
        texture = dr_renderer.depth_target.ref,
        load_op = .LOAD,
        store_op = .STORE,
        stencil_load_op = .CLEAR,
        stencil_store_op = .STORE,
        clear_stencil = 0
      };
      pass := SDL_BeginGPURenderPass(command_buffer, null, 0, *depth_target);
      viewport := SDL_GPUViewport.{
        0, 0,
        xx dr_renderer.resolution.x, xx dr_renderer.resolution.y,
        dr_min_stencil_depth, dr_max_stencil_depth,
      };
      SDL_SetGPUViewport(pass, *viewport);
      SDL_BindGPUGraphicsPipeline(pass, dr_renderer.pipeline_shadow.ref_base);
      SDL_PushGPUVertexUniformData(command_buffer, 0, *vert_constants, xx size_of(DR_MeshVertConstants));
      binding_vertex := SDL_GPUBufferBinding.{dr_renderer.shadow_vertex_buffer.ref, 0};
      SDL_BindGPUVertexBuffers(pass, 0, *binding_vertex, 1);
      for shadow_volumes
      {
        SDL_DrawGPUPrimitives(pass,
          xx it.vert_count,
          xx it.instance_count,
          xx it.base_vert,
          xx it.base_instance,
        );
      }
      SDL_EndGPURenderPass(pass);
      for each_element(dr_renderer.draw_meshes)
      {
        colour_target := SDL_GPUColorTargetInfo.{
          texture = colour_target_texture,
          load_op = .LOAD,
          store_op = .STORE
        };
        depth_target = SDL_GPUDepthStencilTargetInfo.{
          texture = dr_renderer.depth_target.ref,
          load_op = .LOAD,
          store_op = .STORE
        };
        vert_constants.light_pos = light.position;
        frag_constants: DR_MeshFragConstants;
        frag_constants.light = light;
        SDL_PushGPUVertexUniformData(command_buffer, 0, *vert_constants, xx size_of(DR_MeshVertConstants));
        SDL_PushGPUFragmentUniformData(command_buffer, 0, *frag_constants, xx size_of(DR_MeshFragConstants));
        sampler := dr_get_sampler(.LINEAR, .LINEAR, .REPEAT);
        main_pass := SDL_BeginGPURenderPass(command_buffer, *colour_target, 1, *depth_target);
        SDL_BindGPUGraphicsPipeline(main_pass, dr_renderer.pipeline_mesh.ref_base);
        SDL_BindGPUVertexStorageBuffers(main_pass, 0, *dr_renderer.mesh_instances.ref, 1);
        binding_vertex = SDL_GPUBufferBinding.{it.vertices.ref, 0};
        binding_index := SDL_GPUBufferBinding.{it.indices.ref, 0};
        SDL_BindGPUVertexBuffers(main_pass, 0, *binding_vertex, 1);
        SDL_BindGPUIndexBuffer(main_pass, *binding_index, ._32BIT);
        binding_textures: [3] SDL_GPUTextureSamplerBinding;
        if it.material
        {
          binding_textures = SDL_GPUTextureSamplerBinding.[
            .{it.material.tex_colour.ref, sampler},
            .{it.material.tex_normal.ref, sampler},
            .{it.material.tex_aorm.ref, sampler},
          ];
        }
        else
        {
          binding_textures = SDL_GPUTextureSamplerBinding.[
            .{dr_renderer.tex_gray.ref, sampler},
            .{dr_renderer.tex_normal.ref, sampler},
            .{dr_renderer.tex_white.ref, sampler}
          ];
        }
        SDL_BindGPUFragmentSamplers(main_pass, 0, binding_textures.data, xx binding_textures.count);
        SDL_DrawGPUIndexedPrimitives(main_pass,
          xx it.index_count,
          xx it.instance_count,
          xx it.base_index,
          xx it.base_vertex,
          xx it.base_instance
        );
        SDL_EndGPURenderPass(main_pass);
      }
    }
    // NOTE(tt): ambient pass
    for each_element(dr_renderer.draw_meshes)
    {
      colour_target := SDL_GPUColorTargetInfo.{
        texture = colour_target_texture,
        load_op = .LOAD,
        store_op = .STORE
      };
      depth_target := SDL_GPUDepthStencilTargetInfo.{
        texture = dr_renderer.depth_target.ref,
        load_op = .LOAD,
        store_op = .STORE
      };
      frag_constants: DR_MeshFragConstants;
      frag_constants.light.type = .Ambient;
      SDL_PushGPUVertexUniformData(command_buffer, 0, *vert_constants, xx size_of(DR_MeshVertConstants));
      SDL_PushGPUFragmentUniformData(command_buffer, 0, *frag_constants, xx size_of(DR_MeshFragConstants));
      sampler := dr_get_sampler(.LINEAR, .LINEAR, .REPEAT);
      main_pass := SDL_BeginGPURenderPass(command_buffer, *colour_target, 1, *depth_target);
      SDL_BindGPUGraphicsPipeline(main_pass, dr_renderer.pipeline_ambient.ref_base);
      SDL_BindGPUVertexStorageBuffers(main_pass, 0, *dr_renderer.mesh_instances.ref, 1);
      binding_vertex := SDL_GPUBufferBinding.{it.vertices.ref, 0};
      binding_index := SDL_GPUBufferBinding.{it.indices.ref, 0};
      SDL_BindGPUVertexBuffers(main_pass, 0, *binding_vertex, 1);
      SDL_BindGPUIndexBuffer(main_pass, *binding_index, ._32BIT);
      binding_textures: [3] SDL_GPUTextureSamplerBinding;
      if it.material
      {
        binding_textures = SDL_GPUTextureSamplerBinding.[
          .{it.material.tex_colour.ref, sampler},
          .{it.material.tex_normal.ref, sampler},
          .{it.material.tex_aorm.ref, sampler},
        ];
      }
      else
      {
        binding_textures = SDL_GPUTextureSamplerBinding.[
          .{dr_renderer.tex_gray.ref, sampler},
          .{dr_renderer.tex_normal.ref, sampler},
          .{dr_renderer.tex_white.ref, sampler}
        ];
      }
      SDL_BindGPUFragmentSamplers(main_pass, 0, binding_textures.data, xx binding_textures.count);
      SDL_DrawGPUIndexedPrimitives(main_pass,
        xx it.index_count,
        xx it.instance_count,
        xx it.base_index,
        xx it.base_vertex,
        xx it.base_instance
      );
      SDL_EndGPURenderPass(main_pass);
    }
    colour_target := SDL_GPUColorTargetInfo.{
      texture = colour_target_texture,
      load_op = .LOAD,
      store_op = .STORE
    };
    depth_target := SDL_GPUDepthStencilTargetInfo.{
      texture = dr_renderer.depth_target.ref,
      load_op = .LOAD,
      store_op = .STORE
    };
    lines_pass := SDL_BeginGPURenderPass(command_buffer, *colour_target, 1, *depth_target);
    SDL_BindGPUGraphicsPipeline(lines_pass, dr_renderer.pipeline_lines.ref_base);
    SDL_PushGPUVertexUniformData(command_buffer, 0, *vert_constants, xx size_of(DR_MeshVertConstants));
    SDL_BindGPUVertexStorageBuffers(lines_pass, 0, *dr_renderer.line_instances.ref, 1);
    for each_element(dr_renderer.draw_lines)
    {
      binding_vertex := SDL_GPUBufferBinding.{dr_renderer.mesh_line.vertices.ref, 0};
      binding_index := SDL_GPUBufferBinding.{dr_renderer.mesh_line.indices.ref, 0};
      SDL_BindGPUVertexBuffers(lines_pass, 0, *binding_vertex, 1);
      SDL_BindGPUIndexBuffer(lines_pass, *binding_index, ._32BIT);
      SDL_DrawGPUIndexedPrimitives(lines_pass, xx dr_renderer.mesh_line.index_count, xx it.instance_count, 0, 0, xx it.base_instance);
    }
    SDL_EndGPURenderPass(lines_pass);
    colour_target = SDL_GPUColorTargetInfo.{
      texture = colour_target_texture,
      load_op = .LOAD,
      store_op = .STORE
    };
    quads_pass := SDL_BeginGPURenderPass(command_buffer, *colour_target, 1, null);
    SDL_BindGPUGraphicsPipeline(quads_pass, dr_renderer.pipeline_quads.ref_base);
    SDL_PushGPUVertexUniformData(command_buffer, 0, *vert_constants, xx size_of(DR_MeshVertConstants));
    SDL_BindGPUVertexStorageBuffers(quads_pass, 0, *dr_renderer.quad_instances.ref, 1);
    for each_element(dr_renderer.draw_quads)
    {
      binding_vertex := SDL_GPUBufferBinding.{dr_renderer.mesh_quad.vertices.ref, 0};
      binding_index := SDL_GPUBufferBinding.{dr_renderer.mesh_quad.indices.ref, 0};
      SDL_BindGPUVertexBuffers(quads_pass, 0, *binding_vertex, 1);
      SDL_BindGPUIndexBuffer(quads_pass, *binding_index, ._32BIT);
      SDL_DrawGPUIndexedPrimitives(quads_pass, xx dr_renderer.mesh_quad.index_count, xx it.instance_count, 0, 0, xx it.base_instance);
    }
    SDL_EndGPURenderPass(quads_pass);
    ImGui.Render();
    draw_data := ImGui.GetDrawData();
    ImGui_ImplSDLGPU3_PrepareDrawData(draw_data, command_buffer);
    target_info: SDL_GPUColorTargetInfo;
    target_info.texture = colour_target_texture;
    if dr_msaa_enabled
    {
      target_info.resolve_texture = dr_renderer.resolve_target.ref;
      target_info.store_op = .RESOLVE_AND_STORE;
    }
    else
    {
      target_info.store_op = .STORE;
    }
    target_info.load_op = .LOAD;
    render_pass := SDL_BeginGPURenderPass(command_buffer, *target_info, 1, null);
    ImGui_ImplSDLGPU3_RenderDrawData(draw_data, command_buffer, render_pass);
    SDL_EndGPURenderPass(render_pass);
    if dr_msaa_enabled
    {
      info: SDL_GPUBlitInfo;
      info.source.texture = dr_renderer.resolve_target.ref;
      info.source.w = xx dr_renderer.resolution.x;
      info.source.h = xx dr_renderer.resolution.y;
      info.destination.texture = swap_texture;
      info.destination.w = swap_width;
      info.destination.h = swap_height;
      SDL_BlitGPUTexture(command_buffer, *info);
    }
    SDL_SubmitGPUCommandBuffer(command_buffer);
  }
  else
  {
    SDL_CancelGPUCommandBuffer(command_buffer);
  }
  ImGui.EndFrame();
  dr_renderer.draw_meshes = null;
  dr_renderer.draw_mesh_count = 0;
  dr_renderer.draw_lines = null;
  dr_renderer.draw_quads = null;
  dr_reset_buffer(dr_renderer.mesh_instances);
  dr_reset_buffer(dr_renderer.line_instances);
  dr_reset_buffer(dr_renderer.quad_instances);
  dr_reset_buffer(dr_renderer.shadow_vertex_buffer);
  array_reset(*dr_renderer.lights);
  arena_reset(dr_renderer.draw_mesh_arena);
}

dr_create_storage_buffer :: (name: string, size: int) -> *SDL_GPUBuffer
{
	info: SDL_GPUBufferCreateInfo;
	info.usage = SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ;
	info.size = xx size;
	buffer := SDL_CreateGPUBuffer(dr_renderer.device, *info);
	return buffer;
}

dr_create_transfer_buffer :: (size: int) -> *SDL_GPUTransferBuffer
{
	info: SDL_GPUTransferBufferCreateInfo;
	info.usage = .UPLOAD;
	info.size = xx size;
	buffer := SDL_CreateGPUTransferBuffer(dr_renderer.device, *info);
	return buffer;
}

#scope_file

dr_get_shader :: (name: string) -> *DR_Shader
{
  source_file := str_format(temp_arena, "%/%.hlsl", dr_shaders_dir, name);
  allocator := arena_allocator(temp_arena);
  source_ok, source := read_entire_file(source_file,, allocator);
  needs_compile := true;
  if source_ok
  {
    hash := get_hash(source);
    for each_element(dr_renderer.shaders)
    {
      if name == it.name
      {
        path := str_format(temp_arena, "%/%.shader_binary", dr_shaders_dir, name);
        file, ok := file_open(path);
        if ok
        {
          header: DR_ShaderHeader;
          file_read(file, *header, size_of(DR_ShaderHeader));
          file_close(*file);
          if header.hash == hash
          {
            needs_compile = false;
          }
        }
        break;
      }
    }
  }
  if needs_compile
  {
    dr_compile_shader(name);
  }
  file, ok := file_open(str_format(temp_arena, "%/%.shader_binary", dr_shaders_dir, name));
  if ok
  {
    header: DR_ShaderHeader;
    file_read(*file, *header, size_of(DR_ShaderHeader));
    vert_spirv := push(temp_arena, xx header.vert_size);
    frag_spirv := push(temp_arena, xx header.frag_size);
    file_read(*file, vert_spirv, header.vert_size);
    file_read(*file, frag_spirv, header.frag_size);
    file_close(*file);
    vert_name := str_format(temp_arena, "%_vert", name);
    frag_name := str_format(temp_arena, "%_frag", name);
    vert := dr_create_shader(vert_name, .VERTEX, header, vert_spirv);
    frag := dr_create_shader(frag_name, .FRAGMENT, header, frag_spirv);
    shader := push_struct(dr_renderer.arena, DR_Shader);
    shader.vert_meta = header.vert_meta;
    shader.frag_meta = header.frag_meta;
    shader.vert = vert;
    shader.frag = frag;
    shader.name = name;
    stack_push(*dr_renderer.shaders, shader);
    return shader;
  }
  return null;
}

DR_ShaderHeader :: struct
{
  hash: u32;
  vert_meta: SDL_ShaderCross_GraphicsShaderResourceInfo;
  frag_meta: SDL_ShaderCross_GraphicsShaderResourceInfo;
  vert_size: int;
  frag_size: int;
}

dr_compile_shader :: (name: string)
{
  allocator := arena_allocator(temp_arena);
  code, ok := read_entire_file(str_format(temp_arena, "%/%.hlsl", dr_shaders_dir, name),, allocator);
  if ok
  {
    DR_ShaderCompilerInfo :: struct
    {
      stage: SDL_ShaderCross_ShaderStage;
      entry: string;
      spirv: *void;
      spirv_size: int;
      meta: SDL_ShaderCross_GraphicsShaderResourceInfo;
    }
    infos := DR_ShaderCompilerInfo.[
      .{stage = .VERTEX, entry = "vert_main"},
      .{stage = .FRAGMENT, entry = "frag_main"},
    ];
    for * infos
    {
      hlsl_info: SDL_ShaderCross_HLSL_Info;
      hlsl_info.source = cstr_from_str(temp_arena, code);
      hlsl_info.entrypoint = cstr_from_str(temp_arena, it.entry);
      hlsl_info.shader_stage = it.stage;
      props := SDL_CreateProperties();
      // SDL_SetBooleanProperty(props, SDL_SHADERCROSS_PROP_SHADER_DEBUG_ENABLE_BOOLEAN, true);
      SDL_SetBooleanProperty(props, SDL_SHADERCROSS_PROP_SHADER_CULL_UNUSED_BINDINGS_BOOLEAN, true);
      SDL_SetStringProperty(props, SDL_SHADERCROSS_PROP_SHADER_DEBUG_NAME_STRING, cstr_from_str(temp_arena, name));
      hlsl_info.props = props;
      spirv: *void;
      spirv_size: u64;
      spirv = SDL_ShaderCross_CompileSPIRVFromHLSL(*hlsl_info, *spirv_size);
      if (spirv)
      {
        spirv_info: SDL_ShaderCross_SPIRV_Info;
        spirv_info.bytecode = spirv;
        spirv_info.bytecode_size = spirv_size;
        spirv_info.entrypoint = hlsl_info.entrypoint;
        spirv_info.shader_stage = it.stage;
        meta := SDL_ShaderCross_ReflectGraphicsSPIRV(spirv, spirv_size, props);
        it.spirv = spirv;
        it.spirv_size = xx spirv_size;
        it.meta = meta.resource_info;
      }
      else
      {
        print("%", sys_get_sdl_error());
      }
    }
    hash := get_hash(code);
    header: DR_ShaderHeader;
    header.hash = hash;
    header.vert_meta = infos[0].meta;
    header.vert_size = infos[0].spirv_size;
    header.frag_meta = infos[1].meta;
    header.frag_size = infos[1].spirv_size;
    file := file_open(str_format(temp_arena, "%/%.shader_binary", dr_shaders_dir, name), true);
    file_write(*file, *header, size_of(DR_ShaderHeader));
    file_write(*file, infos[0].spirv, infos[0].spirv_size);
    file_write(*file, infos[1].spirv, infos[1].spirv_size);
    file_close(*file);
  }
}

dr_create_shader :: (name: string, stage: SDL_GPUShaderStage, header: DR_ShaderHeader, spirv: *void) -> *SDL_GPUShader
{
  spirv_size := ifx stage == .VERTEX then header.vert_size else header.frag_size;
  meta := ifx stage == .VERTEX then header.vert_meta else header.frag_meta;
  entry := ifx stage == .VERTEX then "vert_main" else "frag_main";
  props := SDL_CreateProperties();
  SDL_SetStringProperty(props, SDL_PROP_GPU_SHADER_CREATE_NAME_STRING, cstr_from_str(temp_arena, name));
  create_info: SDL_GPUShaderCreateInfo;
  create_info.code_size = xx spirv_size;
  create_info.code = spirv;
  create_info.entrypoint = cstr_from_str(temp_arena, entry);
  create_info.format = SDL_GPU_SHADERFORMAT_SPIRV;
  create_info.stage = stage;
  create_info.num_samplers = meta.num_samplers;
  create_info.num_storage_textures = meta.num_storage_textures;
  create_info.num_storage_buffers = meta.num_storage_buffers;
  create_info.num_uniform_buffers = meta.num_uniform_buffers;
  create_info.props = props;
  shader := SDL_CreateGPUShader(dr_renderer.device, *create_info);
  return shader;
}