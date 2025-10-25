
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

test :: ()
{
  arena := create_arena();
  map_entities := parse_map(arena, "levels/unnamed.map");
  for map_entities
  {
    if it.planes.count > 0
    {
      mesh := mesh_from_brush(arena, it);
    }
  }
}

// U = dot(position, Uaxis) / Uscale + Uoffset
// V = dot(position, Vaxis) / Vscale + Voffset

MapPlane :: struct
{
  p0, p1, p2: V3;
  u: Plane;
  v: Plane;
  material: string;
  texture_rotation: float;
  tex_coord_scale: V2;
}

MapEntity :: struct
{
  class: string;
  origin: V3;
  planes: [..] MapPlane;
}

parse_map :: (arena: *Arena, path: string) -> [] MapEntity
{
  file_data, ok := read_entire_file(path);
  _, lines := file_to_array("", true, true, false, true, file_data);
  entities := push_array(arena, MapEntity);
  first := true;
  entity: MapEntity;
  for lines
  {
    token, right := break_by_spaces(it);
    if token ==
    {
      case "entity";
        if !first
        {
          array_add(*entities, entity);
        }
        entity = MapEntity.{};
        entity.class = right;
        entity.planes = push_array(arena, MapPlane);
        first = false;
      case "origin";
        scan2(right, "%f %f %f", *entity.origin.x, *entity.origin.y, *entity.origin.z);
      case "plane";
        plane: MapPlane;
        p0:, right = break_by_spaces(right);
        scan2(p0, "(%,%,%)", *plane.p0.x, *plane.p0.y, *plane.p0.z);
        p1:, right = break_by_spaces(right);
        scan2(p1, "(%,%,%)", *plane.p1.x, *plane.p1.y, *plane.p1.z);
        p2:, right = break_by_spaces(right);
        scan2(p2, "(%,%,%)", *plane.p2.x, *plane.p2.y, *plane.p2.z);
        u:, right = break_by_spaces(right);
        scan2(u, "(%,%,%,%)", *plane.u.normal.x, *plane.u.normal.y, *plane.u.normal.z, *plane.u.d);
        v:, right = break_by_spaces(right);
        scan2(v, "(%,%,%,%)", *plane.v.normal.x, *plane.v.normal.y, *plane.v.normal.z, *plane.v.d);
        mat:, right = break_by_spaces(right);
        plane.material = mat;
        rot:, right = break_by_spaces(right);
        plane.texture_rotation = parse_float(*rot);
        tex_scale:, right = break_by_spaces(right);
        scan2(tex_scale, "(%,%)", *plane.tex_coord_scale.x, *plane.tex_coord_scale.y);
        array_add(*entity.planes, plane);
    }
  }
  return entities;
}

mesh_from_brush :: (arena: *Arena, entity: MapEntity) -> *DR_Mesh
{
  planes := push_view(arena, Plane, entity.planes.count);
  for entity.planes
  {
    planes[it_index] = plane_from_points(it.p0, it.p1, it.p2);
  }
  points := push_array(arena, V3);
  for i: 0..planes.count-3
  {
    for j: i+1..planes.count-2
    {
      for k: j+1..planes.count-1
      {
        hit, pos := intersect_planes(planes[i], planes[j], planes[k]);
        if hit && point_inside_brush(pos, planes)
        {
          array_add(*points, pos);
        }
      }
    }
  }
  verts := push_view(arena, DR_MeshVertex, points.count);
  for points
  {
    print("%\n",it);
    verts[it_index].position = it;
  }
  mesh := dr_create_mesh("brush", verts, .[]);
  return mesh;
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
      swap_composition := SDL_GPUSwapchainComposition.SDR;
      present_mode := SDL_GPUPresentMode.VSYNC;
      if SDL_WindowSupportsGPUSwapchainComposition(device, window, .SDR_LINEAR)
      {
        swap_composition = .SDR_LINEAR;
      }
      if SDL_WindowSupportsGPUPresentMode(device, window, .MAILBOX)
      {
        present_mode = .MAILBOX;
      }
      SDL_SetGPUSwapchainParameters(device, window, swap_composition, present_mode);
      imgui_context := ImGui.CreateContext();
      ImGui.SetCurrentContext(imgui_context);
      ImGui_ImplSDL3_InitForSDLGPU(window);
      imgui_info: ImGui_ImplSDLGPU3_InitInfo;
      imgui_info.Device = device;
      imgui_info.ColorTargetFormat = SDL_GetGPUSwapchainTextureFormat(device, window);
      imgui_info.MSAASamples = dr_msaa_sample_count;
      ImGui_ImplSDLGPU3_Init(*imgui_info);
      imgui_io = ImGui.GetIO();
      SDL_SetWindowRelativeMouseMode(window, true);
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