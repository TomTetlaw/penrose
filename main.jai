
SDL_WINDOW_VULKAN: u64: 0x0000000010000000;
SDL_WINDOW_BORDERLESS: u64: 0x0000000000000010;
SDL_WINDOW_HIDDEN: u64: 0x0000000000000008;
SDL_WINDOWPOS_CENTERED: s32: 0x2FFF0000;

game_paused := true;
game_delta: float;
game_time: float;
real_delta: float;
real_time: float;
frame_num: int;

set_fps_mode :: (enabled: bool)
{
  SDL_SetWindowRelativeMouseMode(window, enabled);
}

quantize :: (v: V3, eps: float) -> [3] int {
  return .[cast(int)(v.x / eps), cast(int)(v.y / eps), cast(int)(v.z / eps)];
}

build_map_mesh :: (arena: *Arena, path: string) -> [] *DR_Mesh
{
  file, ok := file_open(path);
  defer file_close(*file);
  if ok
  {
    mesh_count: u32;
    file_read(*file, *mesh_count, size_of(u32));
    meshes := push_view(arena, *DR_Mesh, mesh_count);
    for 0..mesh_count-1
    {
      vert_count: u32;
      file_read(*file, *vert_count, size_of(u32));
      index_count: u32;
      file_read(*file, *index_count, size_of(u32));
      verts := push_view(temp_arena, DR_MeshVertex, vert_count);
      file_read(*file, verts.data, size_of(DR_MeshVertex) * vert_count);
      indices := push_view(temp_arena, u32, index_count);
      file_read(*file, indices.data, size_of(u32) * index_count);
      meshes[it] = dr_create_mesh("test", verts, indices);
    }
    return meshes;
  }
  return .[];
}

EntityType :: enum
{
  PlayerStart;
  Light;
}

Entity :: struct
{
  next: *Entity;
  type: EntityType;
  name: string;
  position: V3;
  light_type: LightType;
  light_inner_rad := 5.0;
  light_outer_rad := 25.0;
  light_brightness := 2.0;
}

read_map_entities :: (arena: *Arena, path: string) -> *Entity
{
  entities: *Entity;
  file_data, ok := read_entire_file(path);
  _, lines := file_to_array("", true, true, false, true, file_data);
  for lines
  {
    entity := push_struct(arena, Entity);
    stack_push(*entities, entity);
    terms := push_array(arena, string);
    line := it;
    while line
    {
      term:, line = break_by_spaces(line);
      array_add(*terms, term);
    }
    cursor := 0;
    while true
    {
      if cursor >= terms.count
      {
        break;
      }
      term := terms[cursor];
      cursor += 1;
      if term ==
      {
        case "classname";
        {
          name := terms[cursor];
          cursor += 1;
          if name ==
          {
            case "info_player_start";
              entity.type = .PlayerStart;
            case "light";
              entity.type = .Light;
          }
        }
        case "origin";
        {
          entity.position.x = parse_float(*terms[cursor + 0])*.1;
          entity.position.y = parse_float(*terms[cursor + 1])*.1;
          entity.position.z = parse_float(*terms[cursor + 2])*.1;
          cursor += 3;
        }
        case "name";
        {
          entity.name = str_copy(arena, terms[cursor]);
          cursor += 1;
        }
        case "light_type";
        {
          entity.light_type = cast(LightType) parse_int(*terms[cursor]);
          cursor += 1;
        }
        case "light_brightness";
        {
          entity.light_brightness = parse_float(*terms[cursor]);
          cursor += 1;
        }
        case "light_inner_radius";
        {
          entity.light_inner_rad = parse_float(*terms[cursor]);
          cursor += 1;
        }
        case "light_outer_radius";
        {
          entity.light_outer_rad = parse_float(*terms[cursor]);
          cursor += 1;
        }
      }
    }
  }
  return entities;
}

meshes: [] *DR_Mesh;
entities: *Entity;

test :: ()
{
  arena := create_arena();
  meshes = build_map_mesh(arena, "levels/unnamed.map_geo");
  entities = read_map_entities(arena, "levels/unnamed.entities.txt");
}

main :: ()
{
  SDL_Init(SDL_INIT_VIDEO);
  SDL_ShaderCross_Init();
  window_flags := SDL_WINDOW_VULKAN | SDL_WINDOW_BORDERLESS | SDL_WINDOW_HIDDEN;
  window = SDL_CreateWindow("Walker", 1, 1, window_flags);
  if window
  {
    display_count: s32;
    displays := SDL_GetDisplays(*display_count);
    display_mode := SDL_GetDesktopDisplayMode(displays[0]);
    window_pos := SDL_WINDOWPOS_CENTERED;
    resolution := V2.{xx display_mode.w, xx display_mode.h} * .9;
    SDL_SetWindowSize(window, xx resolution.x, xx resolution.y);
    SDL_SetWindowPosition(window, window_pos, window_pos);
    device := SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, true, "vulkan");
    if device
    {
      SDL_ClaimWindowForGPUDevice(device, window);
      if SDL_WindowSupportsGPUSwapchainComposition(device, window, .SDR_LINEAR)
      {
        SDL_SetGPUSwapchainParameters(device, window, .SDR_LINEAR, .MAILBOX);
      }
      imgui_context := ImGui.CreateContext();
      ImGui.SetCurrentContext(imgui_context);
      ImGui_ImplSDL3_InitForSDLGPU(window);
      imgui_info: ImGui_ImplSDLGPU3_InitInfo;
      imgui_info.Device = device;
      imgui_info.ColorTargetFormat = SDL_GetGPUSwapchainTextureFormat(device, window);
      imgui_info.MSAASamples = dr_msaa_sample_count;
      ImGui_ImplSDLGPU3_Init(*imgui_info);
      imgui_io = ImGui.GetIO();
      SDL_ShowWindow(window);
      base_init();
      dr_init(window, device, resolution);
      in_init();
      test();
      camera: CAM_Editor;
      while main_loop
      {
        update_time();
        ImGui_ImplSDLGPU3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui.NewFrame();
        event: SDL_Event;
        events: [..] SDL_Event;
        events.allocator = arena_allocator(temp_arena);
        while SDL_PollEvent(*event)
        {
          ImGui_ImplSDL3_ProcessEvent(*event);
          array_add(*events, event);
        }
        for events
        {
          type := cast(SDL_EventType) it.type;
          if type ==
          {
            case .SDL_EVENT_QUIT;
            {
              penrose_quit();
            }
          }
        }
        // NOTE(tt): collect input
        in_process_events(events, !imgui_io.WantCaptureMouse, !imgui_io.WantTextInput);
        if in_key_pressed(.Escape)
        {
          exit(0);
        }
        // NOTE(tt): update
        cam_editor_update(*camera);
        // NOTE(tt): draw
        for each_element(entities)
        {
          if it.type == .Light
          {
            light: DR_LightInfo;
            light.position = it.position;
            light.colour = v3(1.0);
            light.brightness = it.light_brightness;
            light.inner_radius = it.light_inner_rad;
            light.outer_radius = it.light_outer_rad;
            dr_add_light(light);
          }
        }
        for meshes
        {
          dr_draw_mesh(it, .[.{transform=mat_identity(),colour=v4(1.0)}], dr_get_material("grate"));
        }
        dr_draw_frame(*camera);
        arena_reset(temp_arena);
        #if PENROSE_PROFILE
        {
          ___tracy_emit_frame_mark("");
        }
      }
    }
  }
}

penrose_quit :: ()
{
  main_loop = false;
}

#scope_file

main_loop := true;
imgui_io: *ImGuiIO;
window: *SDL_Window;
old_time: Apollo_Time;
update_time :: ()
{
  now := current_time_monotonic();
  diff := now - old_time;
  old_time = now;
  real_time = cast(float) to_float64_seconds(now);
  real_delta = cast(float) to_float64_seconds(diff);
  if game_paused
  {
    game_delta = 0.0;
  }
  else
  {
    game_time += real_delta;
    game_delta = real_delta;
  }
  frame_num += 1;
}

#scope_file