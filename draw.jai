
dr_shaders_dir :: "shaders";
dr_max_mesh_instances :: 1024;

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
  transform: Mat4x4F32;
  colour: Vec4F32;
}

DR_MeshConstants :: struct
{
  world_to_view: Mat4x4F32;
  view_to_clip: Mat4x4F32;
}

DR_DrawMesh :: struct
{
  next: *DR_DrawMesh;
  instances: DR_BufferRef;
  vertices: DR_BufferRef;
  indices: DR_BufferRef;
  base_instance: int;
  base_vertex: int;
  base_index: int;
  instance_count: int;
  index_count: int;
}

DR_Mesh :: struct
{
  next: *DR_Mesh;
  buffer_vertex: *DR_Buffer;
  buffer_index: *DR_Buffer;
  vertices: DR_BufferRef;
  indices: DR_BufferRef;
}

DR_RendererState :: struct
{
  arena: *Arena;
  device: *SDL_GPUDevice;
  window: *SDL_Window;
  resolution: Vec2F32;
  shaders: *DR_Shader;
  pipelines: *DR_Pipeline;
  buffers: *DR_Buffer;
  meshes: *DR_Mesh;
  mesh_instances: *DR_Buffer;
  shader_mesh: *DR_Shader;
  pipeline_mesh: *DR_Pipeline;
  draw_meshes: *DR_DrawMesh;

  mesh_cube: *DR_Mesh;

  upload_size: int;
  upload_queue: *DR_Upload;
  last_upload: *DR_Upload;
  transfer_buffer: *SDL_GPUTransferBuffer;
  transfer_buffer_size: int;

  sampler_set_size: Vec3U32;
  sampler_set: [] *SDL_GPUSampler;
}

DR_Pipeline :: struct
{
  next: *DR_Pipeline;
  ref_base: *SDL_GPUGraphicsPipeline;
  ref_wireframe: *SDL_GPUGraphicsPipeline;
}

DR_Shader :: struct
{
  next: *DR_Shader;
  name: string;
  vert_meta: SDL_ShaderCross_GraphicsShaderMetadata;
  frag_meta: SDL_ShaderCross_GraphicsShaderMetadata;
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
  // NOTE(tt): for buffers
  buffer: *SDL_GPUBuffer;
  offset: int;
}

DR_MeshVertex :: struct
{
  position: Vec3F32;
  tex_coord: Vec2F32;
  colour: Vec4F32;
  normal: Vec3F32;
  tangent: Vec3F32;
}

dr_renderer: *DR_RendererState;

dr_create_mesh :: (name: string, vertices: [] DR_MeshVertex, indices: [] u32) -> *DR_Mesh
{
  capacity_vertices := size_of(DR_MeshVertex) * vertices.count;
  capacity_indices := size_of(u32) * indices.count;
  mesh := arena_push_struct(dr_renderer.arena, DR_Mesh);
  stack_push(*dr_renderer.meshes, mesh);
  mesh.buffer_vertex = dr_create_vertex_buffer(str_format(arena_temp, "%_vertices", name), capacity_vertices, null, 0, 0);
  mesh.buffer_index = dr_create_index_buffer(str_format(arena_temp, "%_indices", name), capacity_indices, null, 0, 0);
  mesh.vertices = dr_write_buffer(mesh.buffer_vertex, vertices.data, capacity_vertices);
  mesh.indices = dr_write_buffer(mesh.buffer_index, indices.data, capacity_indices);
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
	  SDL_SetGPUBufferName(dr_renderer.device, ref, cstr_from_str(arena_temp, name));
	  buffer := arena_push_struct(dr_renderer.arena, DR_Buffer);
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
    dr_queue_buffer_upload(buffer, data, size, 0);
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

dr_draw_mesh :: (arena: *Arena, instances: [] DR_MeshInstance, vertices: DR_BufferRef, indices: DR_BufferRef)
{
  draw_mesh := arena_push_struct(arena, DR_DrawMesh);
  stack_push(*dr_renderer.draw_meshes, draw_mesh);
  draw_mesh.instances = dr_write_buffer(dr_renderer.mesh_instances, instances.data, size_of(DR_MeshInstance) * instances.count);
  draw_mesh.vertices = vertices;
  draw_mesh.indices = indices;
  draw_mesh.base_instance = draw_mesh.instances.offset / size_of(DR_MeshInstance);
  draw_mesh.base_vertex = vertices.offset / size_of(DR_MeshVertex);
  draw_mesh.base_index = indices.offset / size_of(u32);
  draw_mesh.instance_count = instances.count;
  draw_mesh.index_count = indices.size / size_of(u32);
}

dr_create_pipeline :: (shader: *DR_Shader, enable_alpha_blend: bool) -> *DR_Pipeline
{
  colour_target_desc: SDL_GPUColorTargetDescription;
  colour_target_desc.format = .B8G8R8A8_UNORM;
  if enable_alpha_blend
  {
    colour_target_desc.blend_state.enable_blend = true;
    colour_target_desc.blend_state.src_color_blendfactor = .ONE;
    colour_target_desc.blend_state.dst_color_blendfactor = .ONE_MINUS_SRC_ALPHA;
    colour_target_desc.blend_state.color_blend_op = .ADD;
    colour_target_desc.blend_state.src_alpha_blendfactor = .ONE;
    colour_target_desc.blend_state.dst_alpha_blendfactor = .ONE_MINUS_SRC_ALPHA;
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
    .{#run offset_of(DR_MeshVertex, "colour"),    .FLOAT4},
    .{#run offset_of(DR_MeshVertex, "normal"),    .FLOAT3},
    .{#run offset_of(DR_MeshVertex, "tangent"),   .FLOAT3},
  ];
  attributes: [vert_attribute_desc.count] SDL_GPUVertexAttribute;
  for * vert_attribute_desc
  {
    attributes[it_index].location = xx it_index;
    attributes[it_index].buffer_slot = 0;
    attributes[it_index].offset = it.offset;
    attributes[it_index].format = it.format;
  }
  vertex_buffer_desc: SDL_GPUVertexBufferDescription;
  vertex_buffer_desc.slot = 0;
  vertex_buffer_desc.pitch = size_of(DR_MeshVertex);
  vertex_buffer_desc.input_rate = .VERTEX;
  info_base: SDL_GPUGraphicsPipelineCreateInfo;
  info_base.vertex_input_state.vertex_buffer_descriptions = *vertex_buffer_desc;
  info_base.vertex_input_state.num_vertex_buffers = 1;
  info_base.vertex_input_state.vertex_attributes = attributes.data;
  info_base.vertex_input_state.num_vertex_attributes = vert_attribute_desc.count;
  info_base.target_info.color_target_descriptions = *colour_target_desc;
  info_base.target_info.num_color_targets = 1;
  info_base.rasterizer_state.cull_mode = .NONE;//FRONT;
  info_base.rasterizer_state.fill_mode = .FILL;
  info_base.primitive_type = .TRIANGLELIST;
  info_base.vertex_shader = shader.vert;
  info_base.fragment_shader = shader.frag;
  ref_base := SDL_CreateGPUGraphicsPipeline(dr_renderer.device, *info_base);
  info_wireframe := info_base;
  info_wireframe.rasterizer_state.fill_mode = .LINE;
  ref_wireframe := SDL_CreateGPUGraphicsPipeline(dr_renderer.device, *info_wireframe);
  pipeline := arena_push_struct(dr_renderer.arena, DR_Pipeline);
  stack_push(*dr_renderer.pipelines, pipeline);
  pipeline.ref_base = ref_wireframe;
  pipeline.ref_wireframe = ref_base;
  return pipeline;
}

dr_create_builtin_meshes :: () {
  cube_vertices := DR_MeshVertex.[
    .{position=v3f32(-0.5,-0.5, 0.5), tex_coord=v2f32(0,0), colour=v4f32(1,1,1,1), normal=v3f32(0,0,1), tangent=v3f32(1,0,0)},
    .{position=v3f32( 0.5,-0.5, 0.5), tex_coord=v2f32(1,0), colour=v4f32(1,1,1,1), normal=v3f32(0,0,1), tangent=v3f32(1,0,0)},
    .{position=v3f32( 0.5, 0.5, 0.5), tex_coord=v2f32(1,1), colour=v4f32(1,1,1,1), normal=v3f32(0,0,1), tangent=v3f32(1,0,0)},
    .{position=v3f32(-0.5, 0.5, 0.5), tex_coord=v2f32(0,1), colour=v4f32(1,1,1,1), normal=v3f32(0,0,1), tangent=v3f32(1,0,0)},
    .{position=v3f32( 0.5,-0.5,-0.5), tex_coord=v2f32(0,0), colour=v4f32(1,1,1,1), normal=v3f32(0,0,-1), tangent=v3f32(-1,0,0)},
    .{position=v3f32(-0.5,-0.5,-0.5), tex_coord=v2f32(1,0), colour=v4f32(1,1,1,1), normal=v3f32(0,0,-1), tangent=v3f32(-1,0,0)},
    .{position=v3f32(-0.5, 0.5,-0.5), tex_coord=v2f32(1,1), colour=v4f32(1,1,1,1), normal=v3f32(0,0,-1), tangent=v3f32(-1,0,0)},
    .{position=v3f32( 0.5, 0.5,-0.5), tex_coord=v2f32(0,1), colour=v4f32(1,1,1,1), normal=v3f32(0,0,-1), tangent=v3f32(-1,0,0)},
    .{position=v3f32(-0.5,-0.5,-0.5), tex_coord=v2f32(0,0), colour=v4f32(1,1,1,1), normal=v3f32(-1,0,0), tangent=v3f32(0,0,-1)},
    .{position=v3f32(-0.5,-0.5, 0.5), tex_coord=v2f32(1,0), colour=v4f32(1,1,1,1), normal=v3f32(-1,0,0), tangent=v3f32(0,0,-1)},
    .{position=v3f32(-0.5, 0.5, 0.5), tex_coord=v2f32(1,1), colour=v4f32(1,1,1,1), normal=v3f32(-1,0,0), tangent=v3f32(0,0,-1)},
    .{position=v3f32(-0.5, 0.5,-0.5), tex_coord=v2f32(0,1), colour=v4f32(1,1,1,1), normal=v3f32(-1,0,0), tangent=v3f32(0,0,-1)},
    .{position=v3f32( 0.5,-0.5, 0.5), tex_coord=v2f32(0,0), colour=v4f32(1,1,1,1), normal=v3f32(1,0,0), tangent=v3f32(0,0,1)},
    .{position=v3f32( 0.5,-0.5,-0.5), tex_coord=v2f32(1,0), colour=v4f32(1,1,1,1), normal=v3f32(1,0,0), tangent=v3f32(0,0,1)},
    .{position=v3f32( 0.5, 0.5,-0.5), tex_coord=v2f32(1,1), colour=v4f32(1,1,1,1), normal=v3f32(1,0,0), tangent=v3f32(0,0,1)},
    .{position=v3f32( 0.5, 0.5, 0.5), tex_coord=v2f32(0,1), colour=v4f32(1,1,1,1), normal=v3f32(1,0,0), tangent=v3f32(0,0,1)},
    .{position=v3f32(-0.5, 0.5, 0.5), tex_coord=v2f32(0,0), colour=v4f32(1,1,1,1), normal=v3f32(0,1,0), tangent=v3f32(1,0,0)},
    .{position=v3f32( 0.5, 0.5, 0.5), tex_coord=v2f32(1,0), colour=v4f32(1,1,1,1), normal=v3f32(0,1,0), tangent=v3f32(1,0,0)},
    .{position=v3f32( 0.5, 0.5,-0.5), tex_coord=v2f32(1,1), colour=v4f32(1,1,1,1), normal=v3f32(0,1,0), tangent=v3f32(1,0,0)},
    .{position=v3f32(-0.5, 0.5,-0.5), tex_coord=v2f32(0,1), colour=v4f32(1,1,1,1), normal=v3f32(0,1,0), tangent=v3f32(1,0,0)},
    .{position=v3f32(-0.5,-0.5,-0.5), tex_coord=v2f32(0,0), colour=v4f32(1,1,1,1), normal=v3f32(0,-1,0), tangent=v3f32(1,0,0)},
    .{position=v3f32( 0.5,-0.5,-0.5), tex_coord=v2f32(1,0), colour=v4f32(1,1,1,1), normal=v3f32(0,-1,0), tangent=v3f32(1,0,0)},
    .{position=v3f32( 0.5,-0.5, 0.5), tex_coord=v2f32(1,1), colour=v4f32(1,1,1,1), normal=v3f32(0,-1,0), tangent=v3f32(1,0,0)},
    .{position=v3f32(-0.5,-0.5, 0.5), tex_coord=v2f32(0,1), colour=v4f32(1,1,1,1), normal=v3f32(0,-1,0), tangent=v3f32(1,0,0)},
  ];
  cube_indices := u32.[
    0,1,2, 0,2,3,
    4,5,6, 4,6,7,
    8,9,10, 8,10,11,
    12,13,14, 12,14,15,
    16,17,18, 16,18,19,
    20,21,22, 20,22,23,
  ];
  dr_renderer.mesh_cube = dr_create_mesh("dr_default_cube", cube_vertices, cube_indices);
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
  dr_renderer.sampler_set_size = Vec3U32.{xx filters.count, xx mip_filters.count, xx address_modes.count};
  dr_renderer.sampler_set = arena_push_view(dr_renderer.arena, *SDL_GPUSampler, filters.count * mip_filters.count * address_modes.count);
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
        sampler := SDL_CreateGPUSampler(dr_renderer.device, *info);
        index := cast(int) filter +
                 cast(int) mip_filter*filters.count +
                 cast(int) address_mode*mip_filters.count;
        dr_renderer.sampler_set[index] = sampler;
      }
    }
  }
}

dr_init :: (window: *SDL_Window, device: *SDL_GPUDevice, resolution: Vec2F32)
{
  arena := mem_create_arena();
  dr_renderer = arena_push_struct(arena, DR_RendererState);
  dr_renderer.arena = arena;
  dr_renderer.window = window;
  dr_renderer.device = device;
  dr_renderer.resolution = resolution;
  dr_renderer.shader_mesh = dr_get_shader("mesh");
  dr_renderer.pipeline_mesh = dr_create_pipeline(dr_renderer.shader_mesh, true);
  dr_renderer.mesh_instances = dr_create_storage_buffer("mesh_instances", size_of(DR_MeshInstance) * dr_max_mesh_instances, null, 0, 0);
  dr_create_builtin_meshes();
  dr_create_sampler_set();
}

dr_get_sampler :: (filter: SDL_GPUFilter, mip_filter: SDL_GPUSamplerMipmapMode, address_mode: SDL_GPUSamplerAddressMode) -> *SDL_GPUSampler
{
  index := cast(int) filter +
    cast(int) mip_filter*dr_renderer.sampler_set_size.x +
    cast(int) address_mode*dr_renderer.sampler_set_size.y;
  return dr_renderer.sampler_set[index];
}

dr_queue_buffer_upload :: (dest: *DR_Buffer, data: *void, size: int, offset: int)
{
  upload := arena_push_struct(arena_temp, DR_Upload);
  queue_push(*dr_renderer.upload_queue, *dr_renderer.last_upload, upload);
  upload.size = size;
  upload.data = arena_push_copy(arena_temp, data, xx size);
  upload.offset = offset;
  upload.buffer = dest.ref;
  dr_renderer.upload_size += size;
}

dr_queue_texture_upload :: (dest: *SDL_GPUTexture, data: *void, width: int, height: int, stride: int, x := 0, y := 0)
{
  upload := arena_push_struct(arena_temp, DR_Upload);
  queue_push(*dr_renderer.upload_queue, *dr_renderer.last_upload, upload);
  upload.size = width * height * stride;
  upload.width = width;
  upload.height = height;
  upload.data = arena_push_copy(arena_temp, data, xx upload.size);
  upload.x = x;
  upload.y = y;
  upload.texture = dest;
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
  dr_renderer.upload_size = 0;
  queue_reset(*dr_renderer.upload_queue, *dr_renderer.last_upload);
}

dr_draw_frame :: ()
{
  command_buffer := SDL_AcquireGPUCommandBuffer(dr_renderer.device);
  swap_texture: *SDL_GPUTexture;
  SDL_WaitAndAcquireGPUSwapchainTexture(command_buffer, dr_renderer.window, *swap_texture, null, null);
  if (swap_texture)
  {
    instances :=  DR_MeshInstance.[.{mat4x4f32_translate(v3f32(10, 0, 0)), v4f32(1, 0, 0, 1)}];
    vertices := dr_renderer.mesh_cube.vertices;
    indices := dr_renderer.mesh_cube.indices;
    dr_draw_mesh(arena_temp, instances, vertices, indices);
    dr_process_upload_queue(command_buffer);
    colour_target := SDL_GPUColorTargetInfo.{
      texture = swap_texture,
      clear_color = .{.125, .125, .125, 1.0},
      load_op = .CLEAR,
      store_op = .STORE
    };
    constants: DR_MeshConstants;
    constants.world_to_view = mat4x4f32_identity();
    aspect := dr_renderer.resolution.x / dr_renderer.resolution.y;
    constants.view_to_clip = mat4x4f32_perspective(0.125, aspect, 0.1, 1000.0);
    main_pass := SDL_BeginGPURenderPass(command_buffer, *colour_target, 1, null);
    SDL_BindGPUGraphicsPipeline(main_pass, dr_renderer.pipeline_mesh.ref_base);
    SDL_PushGPUVertexUniformData(command_buffer, 0, *constants, xx size_of(DR_MeshConstants));
    SDL_BindGPUVertexStorageBuffers(main_pass, 0, *dr_renderer.mesh_instances.ref, 1);
    for each_element(dr_renderer.draw_meshes)
    {
      binding_vertex := SDL_GPUBufferBinding.{it.vertices.ref, 0};
      binding_index := SDL_GPUBufferBinding.{it.indices.ref, 0};
      SDL_BindGPUVertexBuffers(main_pass, 0, *binding_vertex, 1);
      SDL_BindGPUIndexBuffer(main_pass, *binding_index, ._32BIT);
      SDL_DrawGPUIndexedPrimitives(main_pass, xx it.index_count, xx it.instance_count, xx it.base_index, xx it.base_vertex, xx it.base_instance);
    }
    SDL_EndGPURenderPass(main_pass);
    SDL_SubmitGPUCommandBuffer(command_buffer);
    dr_renderer.draw_meshes = null;
    dr_reset_buffer(dr_renderer.mesh_instances);
  }
  else
  {
    SDL_CancelGPUCommandBuffer(command_buffer);
  }
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

dr_create_texture_2d :: (name: string, width: int, height: int, format: SDL_GPUTextureFormat, usage: u32, generate_mipmaps: bool) -> *SDL_GPUTexture
{
	info: SDL_GPUTextureCreateInfo;
	info.type = ._2D;
	info.format = format;
	info.width = xx width;
	info.height = xx height;
	info.layer_count_or_depth = 1;
	if generate_mipmaps
	{
	 info.num_levels = cast(u32) log(cast(float) max(width, height)) + 1;
  }
  else
  {
    info.num_levels = 1;
  }
	info.usage = usage;
	texture := SDL_CreateGPUTexture(dr_renderer.device, *info);
	SDL_SetGPUTextureName(dr_renderer.device, texture, cstr_from_str(arena_temp, name));
	return texture;
}

#scope_file

dr_get_shader :: (name: string) -> *DR_Shader
{
  source_file := str_format(arena_temp, "%/%.hlsl", dr_shaders_dir, name);
  allocator := arena_allocator(arena_temp);
  source_ok, source := read_entire_file(source_file,, allocator);
  needs_compile := true;
  if source_ok
  {
    hash := get_hash(source);
    for each_element(dr_renderer.shaders)
    {
      if name == it.name
      {
        path := str_format(arena_temp, "%/%.shader_binary", dr_shaders_dir, name);
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
  file, ok := file_open(str_format(arena_temp, "%/%.shader_binary", dr_shaders_dir, name));
  if ok
  {
    header: DR_ShaderHeader;
    file_read(*file, *header, size_of(DR_ShaderHeader));
    vert_spirv := arena_push(arena_temp, xx header.vert_size);
    frag_spirv := arena_push(arena_temp, xx header.frag_size);
    file_read(*file, vert_spirv, header.vert_size);
    file_read(*file, frag_spirv, header.frag_size);
    file_close(*file);
    vert_name := str_format(arena_temp, "%_vert", name);
    frag_name := str_format(arena_temp, "%_frag", name);
    vert := dr_create_shader(vert_name, .VERTEX, header, vert_spirv);
    frag := dr_create_shader(frag_name, .FRAGMENT, header, frag_spirv);
    shader := arena_push_struct(dr_renderer.arena, DR_Shader);
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
  vert_meta: SDL_ShaderCross_GraphicsShaderMetadata;
  frag_meta: SDL_ShaderCross_GraphicsShaderMetadata;
  vert_size: int;
  frag_size: int;
}

dr_compile_shader :: (name: string)
{
  allocator := arena_allocator(arena_temp);
  code, ok := read_entire_file(str_format(arena_temp, "%/%.hlsl", dr_shaders_dir, name),, allocator);
  if ok
  {
    DR_ShaderCompilerInfo :: struct
    {
      stage: SDL_ShaderCross_ShaderStage;
      entry: string;
      spirv: *void;
      spirv_size: int;
      meta: SDL_ShaderCross_GraphicsShaderMetadata;
    }
    infos := DR_ShaderCompilerInfo.[
      .{stage = .VERTEX, entry = "vert_main"},
      .{stage = .FRAGMENT, entry = "frag_main"},
    ];
    for * infos
    {
      hlsl_info: SDL_ShaderCross_HLSL_Info;
      hlsl_info.source = cstr_from_str(arena_temp, code);
      hlsl_info.entrypoint = cstr_from_str(arena_temp, it.entry);
      hlsl_info.shader_stage = it.stage;
      hlsl_info.name = cstr_from_str(arena_temp, name);
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
        meta: SDL_ShaderCross_GraphicsShaderMetadata;
        SDL_ShaderCross_ReflectGraphicsSPIRV(spirv, spirv_size, *meta);
        it.spirv = spirv;
        it.spirv_size = xx spirv_size;
        it.meta = meta;
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
    file := file_open(str_format(arena_temp, "%/%.shader_binary", dr_shaders_dir, name), true);
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
  SDL_SetStringProperty(props, SDL_PROP_GPU_SHADER_CREATE_NAME_STRING, cstr_from_str(arena_temp, name));
  create_info: SDL_GPUShaderCreateInfo;
  create_info.code_size = xx spirv_size;
  create_info.code = spirv;
  create_info.entrypoint = cstr_from_str(arena_temp, entry);
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