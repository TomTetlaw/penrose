
r_shader_directory :: "shaders";
r_max_quad_instances :: 4096;

R_PipelineId :: enum
{
  quads;
}

R_Pipeline :: struct
{
  shader: *R_Shader;
  id: R_PipelineId;
  name: string;
  ref: *SDL_GPUGraphicsPipeline;
  instance_buffer: *SDL_GPUBuffer;
  instance_size: int;
  instance_count: int;
  constants: *void;
  constants_size: int;
}

R_RendererState :: struct
{
  arena: *Arena;
  device: *SDL_GPUDevice;
  window: *SDL_Window;
  resolution: V2_f32;
  shaders: *R_Shader;
  basic_mesh_shader: *R_Shader;

  pipelines: [] R_Pipeline;

  upload_size: int;
  upload_queue: *R_Upload;
  last_upload: *R_Upload;
  transfer_buffer: *SDL_GPUTransferBuffer;
  transfer_buffer_size: int;

  sampler_set_size: V3_u32;
  sampler_set: [] *SDL_GPUSampler;
}

R_Shader :: struct
{
  next: *R_Shader;
  name: string;
  vert_meta: SDL_ShaderCross_GraphicsShaderMetadata;
  frag_meta: SDL_ShaderCross_GraphicsShaderMetadata;
  vert: *SDL_GPUShader;
  frag: *SDL_GPUShader;
}

R_QuadInstance :: struct
{
  position: V2_f32;
  size: V2_f32;
  colour: V4_f32;
}

R_QuadConstants :: struct
{
  screen_to_clip: Mat4x4_f32;
}

R_Upload :: struct
{
  next: *R_Upload;
  size, width, height: int;
  data: *void;
  offset, x, y: int;
  buffer: *SDL_GPUBuffer;
  texture: *SDL_GPUTexture;
}

renderer: *R_RendererState;

r_create_pipeline :: (id: R_PipelineId, shader: *R_Shader) -> *SDL_GPUGraphicsPipeline
{
  if id ==
  {
    case .quads;
    info: SDL_GPUGraphicsPipelineCreateInfo;
    CTD: SDL_GPUColorTargetDescription;
    CTD.format = .B8G8R8A8_UNORM;
    info.target_info.color_target_descriptions = *CTD;
    info.target_info.num_color_targets = 1;
    info.rasterizer_state.cull_mode = .BACK;
    info.primitive_type = .TRIANGLESTRIP;
    info.vertex_shader = shader.vert;
    info.fragment_shader = shader.frag;
    pipeline := SDL_CreateGPUGraphicsPipeline(renderer.device, *info);
    return pipeline;
  }
  return null;
}

r_init :: (window: *SDL_Window, device: *SDL_GPUDevice, resolution: V2_f32)
{
  arena := arena_create();
  renderer = arena_push(arena, R_RendererState);
  renderer.arena = arena;
  renderer.window = window;
  renderer.device = device;
  renderer.resolution = resolution;

  R_PipelineInfo :: struct
  {
    id: R_PipelineId;
    name: string;
    constants_size: int;
    instance_size: int;
    max_instances: int;
  }
  pipeline_count :: #run enum_count(R_PipelineId);
  pipeline_info: [pipeline_count] R_PipelineInfo = .[
    .{.quads, "quads", size_of(R_QuadConstants), size_of(R_QuadInstance), r_max_quad_instances},
  ];
  renderer.pipelines = arena_push_view(renderer.arena, R_Pipeline, pipeline_count);
  for pipeline_info
  {
    pipeline := *renderer.pipelines[it_index];
    pipeline.name = it.name;
    pipeline.id = it.id;
    pipeline.shader = r_get_shader(it.name);
    pipeline.ref = r_create_pipeline(it.id, pipeline.shader);
    pipeline.instance_buffer = r_create_storage_buffer(
      tprint("%_instances", it.name), it.instance_size * it.max_instances
    );
    pipeline.instance_size = it.instance_size;
    pipeline.constants_size = it.constants_size;
    pipeline.constants = arena_alloc(renderer.arena, xx it.constants_size);
  }
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
  renderer.sampler_set_size = V3_u32.{xx filters.count, xx mip_filters.count, xx address_modes.count};
  renderer.sampler_set = arena_push_view(renderer.arena, *SDL_GPUSampler, filters.count * mip_filters.count * address_modes.count);
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
        sampler := SDL_CreateGPUSampler(renderer.device, *info);
        index := cast(int) filter +
                 cast(int) mip_filter*filters.count +
                 cast(int) address_mode*mip_filters.count;
        renderer.sampler_set[index] = sampler;
      }
    }
  }
}

r_get_sampler :: (filter: SDL_GPUFilter, mip_filter: SDL_GPUSamplerMipmapMode, address_mode: SDL_GPUSamplerAddressMode) -> *SDL_GPUSampler
{
  index := cast(int) filter +
    cast(int) mip_filter*renderer.sampler_set_size.x +
    cast(int) address_mode*renderer.sampler_set_size.y;
  return renderer.sampler_set[index];
}

r_draw :: (pipeline_id: R_PipelineId, instances: [] $T)
{
  pipeline := *renderer.pipelines[pipeline_id];
  assert(size_of(T) == pipeline.instance_size);
  r_queue_upload_buffer(
    instances.data,
    instances.count * pipeline.instance_size,
    pipeline.instance_buffer,
    pipeline.instance_count * pipeline.instance_size
  );
  pipeline.instance_count += instances.count;
}

r_queue_upload_buffer :: (data: *void, size: int, dest: *SDL_GPUBuffer, offset := 0)
{
  upload := arena_push(base.frame_arena, R_Upload);
  queue_push(*renderer.upload_queue, *renderer.last_upload, upload);
  upload.size = size;
  upload.data = data;
  upload.offset = offset;
  upload.buffer = dest;
  renderer.upload_size += size;
}

r_queue_upload_texture :: (data: *void, width: int, height: int, stride: int, dest: *SDL_GPUTexture, x := 0, y := 0)
{
  upload := arena_push(base.frame_arena, R_Upload);
  queue_push(*renderer.upload_queue, *renderer.last_upload, upload);
  upload.size = width * height * stride;
  upload.width = width;
  upload.height = height;
  upload.data = data;
  upload.x = x;
  upload.y = y;
  upload.texture = dest;
  renderer.upload_size += upload.size;
}

r_process_upload_queue :: (command_buffer: *SDL_GPUCommandBuffer)
{
  // NOTE(tt): resize transfer buffer if needed
  if (renderer.transfer_buffer_size < renderer.upload_size)
  {
    if (renderer.transfer_buffer)
    {
      SDL_ReleaseGPUTransferBuffer(renderer.device, renderer.transfer_buffer);
    }
    renderer.transfer_buffer = r_create_transfer_buffer(renderer.upload_size);
    renderer.transfer_buffer_size = renderer.upload_size;
  }
  // NOTE(tt): copy data to transfer buffer
  mapping := SDL_MapGPUTransferBuffer(renderer.device, renderer.transfer_buffer, true);
  cursor := 0;
  for each_element(renderer.upload_queue)
  {
    copy(mapping + cursor, it.data, xx it.size);
    cursor += it.size;
  }
  SDL_UnmapGPUTransferBuffer(renderer.device, renderer.transfer_buffer);
  // NOTE(tt): copy from transfer buffer to GPU resources
  cursor = 0;
  copy_pass := SDL_BeginGPUCopyPass(command_buffer);
  for each_element(renderer.upload_queue)
  {
    if it.buffer
    {
      location: SDL_GPUTransferBufferLocation;
      location.transfer_buffer = renderer.transfer_buffer;
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
      location.transfer_buffer = renderer.transfer_buffer;
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
  }
  SDL_EndGPUCopyPass(copy_pass);
  renderer.upload_size = 0;
  queue_reset(*renderer.upload_queue, *renderer.last_upload);
}

r_fill_pipeline_constants :: (id: R_PipelineId, constants: *$T)
{
  pipeline := *renderer.pipelines[id];
  assert(pipeline.constants_size == size_of(T));
  copy(pipeline.constants, constants, size_of(T));
}

r_draw_frame :: ()
{
  quad_constants: R_QuadConstants;
  quad_constants.screen_to_clip = mat4x4_f32_ortho(
    0, xx renderer.resolution.x, xx renderer.resolution.y, 0,
    -1.0, 1.0
  );
  r_fill_pipeline_constants(.quads, *quad_constants);
  command_buffer := SDL_AcquireGPUCommandBuffer(renderer.device);
  swap_texture: *SDL_GPUTexture;
  SDL_WaitAndAcquireGPUSwapchainTexture(command_buffer, renderer.window, *swap_texture, null, null);
  if (swap_texture)
  {
    r_process_upload_queue(command_buffer);
    colour_target := SDL_GPUColorTargetInfo.{
      texture = swap_texture,
      clear_color = .{.125, .125, .125, 1.0},
      load_op = .CLEAR,
      store_op = .STORE
    };
    for * renderer.pipelines
    {
      render_pass := SDL_BeginGPURenderPass(command_buffer, *colour_target, 1, null);
      SDL_BindGPUGraphicsPipeline(render_pass, it.ref);
      SDL_BindGPUVertexStorageBuffers(render_pass, 0, *it.instance_buffer, 1);
      SDL_PushGPUVertexUniformData(command_buffer, 0, it.constants, xx it.constants_size);
      SDL_DrawGPUPrimitives(render_pass, 6, xx it.instance_count, 0, 0);
      SDL_EndGPURenderPass(render_pass);
      SDL_SubmitGPUCommandBuffer(command_buffer);
      it.instance_count = 0;
    }
  }
  else
  {
    SDL_CancelGPUCommandBuffer(command_buffer);
  }
}

r_create_storage_buffer :: (name: string, size: int) -> *SDL_GPUBuffer
{
	info: SDL_GPUBufferCreateInfo;
	info.usage = SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ;
	info.size = xx size;
	buffer := SDL_CreateGPUBuffer(renderer.device, *info);
	return buffer;
}

r_create_transfer_buffer :: (size: int) -> *SDL_GPUTransferBuffer
{
	info: SDL_GPUTransferBufferCreateInfo;
	info.usage = .UPLOAD;
	info.size = xx size;
	buffer := SDL_CreateGPUTransferBuffer(renderer.device, *info);
	return buffer;
}

r_create_texture_2d :: (name: string, width: int, height: int, format: SDL_GPUTextureFormat, usage: u32, generate_mipmaps: bool) -> *SDL_GPUTexture
{
	info: SDL_GPUTextureCreateInfo;
	info.type = .SDL_GPU_TEXTURETYPE_2D;
	info.format = format;
	info.width = xx width;
	info.height = xx height;
	info.layer_count_or_depth = 1;
	info.num_levels = cast(u32) log(cast(float) max(width, height)) + 1;
	info.usage = usage;
	texture := SDL_CreateGPUTexture(renderer.device, *info);
	SDL_SetGPUTextureName(renderer.device, texture, temp_cstr(name));
	return texture;
}

#scope_file

r_get_shader :: (name: string) -> *R_Shader
{
  source_file := tprint("%/%.hlsl", r_shader_directory, name);
  source_ok, source := read_entire_file(source_file,, temp);
  needs_compile := true;
  if source_ok
  {
    hash := get_hash(source);
    for each_element(renderer.shaders)
    {
      if name == it.name
      {
        path := tprint("%/%.spirv", r_shader_directory, name);
        if file_exists(path)
        {
          file, ok := file_open(path);
          if ok
          {
            header: R_ShaderHeader;
            file_read(file, *header, size_of(R_ShaderHeader));
            file_close(*file);
            if header.hash == hash
            {
              needs_compile = false;
            }
          }
        }
        break;
      }
    }
  }
  if needs_compile
  {
    r_compile_shader(name);
  }
  file, ok := file_open(tprint("%/%.spirv", r_shader_directory, name));
  if ok
  {
    header: R_ShaderHeader;
    file_read(*file, *header, size_of(R_ShaderHeader));
    vert_spirv := talloc(header.vert_size);
    frag_spirv := talloc(header.frag_size);
    file_read(*file, vert_spirv, header.vert_size);
    file_read(*file, frag_spirv, header.frag_size);
    file_close(*file);
    vert_name := tprint("%_vert", name);
    frag_name := tprint("%_frag", name);
    vert := r_create_shader(vert_name, .VERTEX, header, vert_spirv);
    frag := r_create_shader(frag_name, .FRAGMENT, header, frag_spirv);
    shader := arena_push(renderer.arena, R_Shader);
    shader.vert_meta = header.vert_meta;
    shader.frag_meta = header.frag_meta;
    shader.vert = vert;
    shader.frag = frag;
    shader.name = name;
    stack_push(*renderer.shaders, shader);
    return shader;
  }
  return null;
}

R_ShaderHeader :: struct
{
  hash: u32;
  vert_meta, frag_meta: SDL_ShaderCross_GraphicsShaderMetadata;
  vert_size, frag_size: int;
}

r_compile_shader :: (name: string)
{
  code, ok := read_entire_file(tprint("%/%.hlsl", r_shader_directory, name),, temp);
  if ok
  {
    R_ShaderCompilerInfo :: struct
    {
      stage: SDL_ShaderCross_ShaderStage;
      entry: string;
      spirv: *void;
      spirv_size: int;
      meta: SDL_ShaderCross_GraphicsShaderMetadata;
    }
    infos := R_ShaderCompilerInfo.[
      .{stage = .VERTEX, entry = "vert_main"},
      .{stage = .FRAGMENT, entry = "frag_main"},
    ];
    for * infos
    {
      hlsl_info: SDL_ShaderCross_HLSL_Info;
      hlsl_info.source = temp_cstr(code);
      hlsl_info.entrypoint = temp_cstr(it.entry);
      hlsl_info.shader_stage = it.stage;
      hlsl_info.name = temp_cstr(name);
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
    header: R_ShaderHeader;
    header.hash = hash;
    header.vert_meta = infos[0].meta;
    header.vert_size = infos[0].spirv_size;
    header.frag_meta = infos[1].meta;
    header.frag_size = infos[1].spirv_size;
    file := file_open(tprint("%/%.spirv", r_shader_directory, name), for_writing=true);
    file_write(*file, *header, size_of(R_ShaderHeader));
    file_write(*file, infos[0].spirv, infos[0].spirv_size);
    file_write(*file, infos[1].spirv, infos[1].spirv_size);
    file_close(*file);
  }
}

r_create_shader :: (name: string, stage: SDL_GPUShaderStage, header: R_ShaderHeader, spirv: *void) -> *SDL_GPUShader
{
  spirv_size := ifx stage == .VERTEX then header.vert_size else header.frag_size;
  meta := ifx stage == .VERTEX then header.vert_meta else header.frag_meta;
  entry := ifx stage == .VERTEX then "vert_main" else "frag_main";
  props := SDL_CreateProperties();
  SDL_SetStringProperty(props, SDL_PROP_GPU_SHADER_CREATE_NAME_STRING, temp_cstr(name));
  create_info: SDL_GPUShaderCreateInfo;
  create_info.code_size = xx spirv_size;
  create_info.code = spirv;
  create_info.entrypoint = temp_cstr(entry);
  create_info.format = SDL_GPU_SHADERFORMAT_SPIRV;
  create_info.stage = stage;
  create_info.num_samplers = meta.num_samplers;
  create_info.num_storage_textures = meta.num_storage_textures;
  create_info.num_storage_buffers = meta.num_storage_buffers;
  create_info.num_uniform_buffers = meta.num_uniform_buffers;
  create_info.props = props;
  shader := SDL_CreateGPUShader(renderer.device, *create_info);
  return shader;
}