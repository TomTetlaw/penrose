
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
    resolution := Vec2F32.{xx display_mode.w, xx display_mode.h} * .9;
    SDL_SetWindowSize(window, xx resolution.x, xx resolution.y);
    SDL_SetWindowPosition(window, window_pos, window_pos);
    device := SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, true, "vulkan");
    if device
    {
      SDL_ClaimWindowForGPUDevice(device, window);
      imgui_context := ImGui.CreateContext();
      ImGui.SetCurrentContext(imgui_context);
      ImGui_ImplSDL3_InitForSDLGPU(window);
      imgui_info: ImGui_ImplSDLGPU3_InitInfo;
      imgui_info.Device = device;
      imgui_info.ColorTargetFormat = SDL_GetGPUSwapchainTextureFormat(device, window);
      ImGui_ImplSDLGPU3_Init(*imgui_info);
      imgui_io = ImGui.GetIO();
      SDL_ShowWindow(window);
      base_init();
      dr_init(window, device, resolution);
      in_init();
      g_init();
      ed_init();
      while main_loop
      {
        update_time();
        ImGui_ImplSDLGPU3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui.NewFrame();
        event: SDL_Event;
        events: [..] SDL_Event;
        events.allocator = arena_allocator(arena_temp);
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
        // NOTE(tt): update
        g_update();
        if g_game_state.mode == .Editor
        {
          ed_update();
        }
        // NOTE(tt): draw
        g_draw(.{include_editor_only = g_game_state.mode == .Editor});
        if g_game_state.mode == .Editor
        {
          ed_draw();
        }
        camera: *CAM_Base;
        if g_game_state.mode == .Editor
        {
          camera = *ed_editor.camera;
        }
        else
        {
          camera = *g_game_state.player.camera;
        }
        frame_start := profile_start();
        dr_draw_frame(camera);
        profile_end("frame", frame_start);
        arena_reset(arena_temp);
      }
    }
  }
  ed_quit();
  dr_quit();
}

penrose_quit :: ()
{
  main_loop = false;
}

#scope_file

profile_start :: () -> F64
{
  now := current_time_monotonic();
  result := to_float64_seconds(now);
  return result;
}

profile_end :: (label: string, start: F64)
{
  now := current_time_monotonic();
  end := to_float64_seconds(now);
  duration := end - start;
  print("% took %s\n", label, duration);
}

main_loop := true;
imgui_io: *ImGuiIO;
window: *SDL_Window;
old_time: Apollo_Time;
update_time :: ()
{
  now := current_time_monotonic();
  diff := now - old_time;
  old_time = now;
  real_time = cast(F32) to_float64_seconds(now);
  real_delta = cast(F32) to_float64_seconds(diff);
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