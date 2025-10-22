
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

build_map_mesh :: (path: string) -> [] *DR_Mesh
{
arena:=create_arena();
  file, ok := file_open(path);
  defer file_close(*file);
  if ok
  {
    mesh_count: u32;
    file_read(*file, *mesh_count, size_of(u32));
    vert_counts := push_view(arena, u32, mesh_count);
    file_read(*file, vert_counts.data, size_of(u32) * mesh_count);
    meshes := push_view(arena, *DR_Mesh, mesh_count);
    for 0..mesh_count-1
    {
      vert_count := vert_counts[it];
      positions := push_view(temp_arena, V3, vert_count);
      file_read(*file, positions.data, size_of(V3) * vert_count);
      verts := push_view(temp_arena, DR_MeshVertex, positions.count);
      indices := push_view(temp_arena, u32, positions.count);
      table: Table(u32, u32);
      table.allocator = arena_allocator(temp_arena);
      pos_count := 0;
      index_count := 0;
      for positions
      {
        q := quantize(it, .0001);
        key := get_hash(q[0], get_hash(q[1], get_hash(q[2])));
        if !table_contains(*table, key)
        {
          verts[pos_count] = .{position=it*.1};
          indices[it_index] = xx pos_count;
          table_add(*table, key, xx pos_count);
          pos_count += 1;
        }
        else
        {
          _, i := table_find_new(*table, key);
          indices[it_index] = i;
        }
        tri_count := indices.count / 3;
        for 0..tri_count-1
        {
          p0 := verts[indices[it*3 + 0]].position;
          p1 := verts[indices[it*3 + 1]].position;
          p2 := verts[indices[it*3 + 2]].position;
          e0 := cross(p1, p0);
          e1 := cross(p2, p0);
          n := normalize(cross(e0, e1));
          t := normalize(cross(n, e0));
          verts[indices[it*3 + 0]].normal = n;
          verts[indices[it*3 + 1]].normal = n;
          verts[indices[it*3 + 2]].normal = n;
          verts[indices[it*3 + 0]].tangent = v4(t, 1.0);
          verts[indices[it*3 + 1]].tangent = v4(t, 1.0);
          verts[indices[it*3 + 2]].tangent = v4(t, 1.0);
          verts[indices[it*3 + 0]].tex_coord = v2(0, 0);
          verts[indices[it*3 + 1]].tex_coord = v2(1, 0);
          verts[indices[it*3 + 2]].tex_coord = v2(0, 1);
        }
      }
      meshes[it] = dr_create_mesh("test", verts, indices);
    }
    return meshes;
  }
  return .[];
}

meshes: [] *DR_Mesh;

test :: ()
{
  meshes = build_map_mesh("levels/unnamed.map_geo");
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
        light: DR_LightInfo;
        light.type = .Point;
        light.position = v3(0,0,50);
        light.colour = v3(1.0);
        light.brightness = 10;
        light.inner_radius = 25;
        light.outer_radius = 200;
        dr_add_light(light);
        light.position = v3(15,0,50);
        dr_add_light(light);
        light.position = v3(0,15,50);
        dr_add_light(light);
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