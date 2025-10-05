
SDL_WINDOW_VULKAN: u64: 0x0000000010000000;
SDL_WINDOW_BORDERLESS: u64: 0x0000000000000010;
SDL_WINDOW_HIDDEN: u64: 0x0000000000000008;
SDL_WINDOWPOS_CENTERED: s32: 0x2FFF0000;

game_paused: bool;
game_delta: float;
game_time: float;
real_delta: float;
real_time: float;
frame_num: int;

main :: ()
{
  SDL_Init(SDL_INIT_VIDEO);
  SDL_ShaderCross_Init();
  window_flags := SDL_WINDOW_VULKAN | SDL_WINDOW_BORDERLESS | SDL_WINDOW_HIDDEN;
  window := SDL_CreateWindow("Walker", 1, 1, window_flags);
  if window
  {
    display_count: s32;
    displays := SDL_GetDisplays(*display_count);
    display_mode := SDL_GetDesktopDisplayMode(displays[0]);
    window_pos := SDL_WINDOWPOS_CENTERED;
    resolution := Vec2F32.{xx display_mode.w, xx display_mode.h} * .75;
    SDL_SetWindowSize(window, xx resolution.x, xx resolution.y);
    SDL_SetWindowPosition(window, window_pos, window_pos);
    device := SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, true, "vulkan");
    if device
    {
      SDL_ClaimWindowForGPUDevice(device, window);
      SDL_ShowWindow(window);
      base_init();
      dr_init(window, device, resolution);
      in_init();
      ed_init();
      g_init();
      material: DR_MeshMaterial;
      material.tex_colour = tex_load("textures/shipping_crates3.tga");
      material.tex_normal = tex_load("textures/shipping_crates3_local.tga");
      material.tex_specular = tex_load("textures/shipping_crates3_s.tga");
      material.specular_shininess = 128;
      while main_loop := true
      {
        event: SDL_Event;
        events: [..] SDL_Event;
        events.allocator = arena_allocator(arena_temp);
        while SDL_PollEvent(*event)
        {
          array_add(*events, event);
        }
        for events
        {
          type := cast(SDL_EventType) it.type;
          if type ==
          {
            case .SDL_EVENT_QUIT; break main_loop;
            case .SDL_EVENT_KEY_DOWN;
            {
              if it.key.scancode == .SDL_SCANCODE_ESCAPE
              {
                break main_loop;
              }
            }
          }
        }
        // NOTE(tt): collect input
        input := in_process_events(events);
        // NOTE(tt): update
        update_time();
        ed_update();
        // NOTE(tt): draw
        ed_draw();
        dr_draw_frame(*ed_editor.camera);
        arena_reset(arena_temp);
      }
    }
  }
}

#scope_file

old_time: Apollo_Time;
update_time :: ()
{
  now := current_time_monotonic();
  diff := now - old_time;
  old_time = now;
  delta := cast(F32) to_float64_seconds(diff);
  real_time += delta;
  real_delta = delta;
  if game_paused
  {
    game_delta = 0.0;
  }
  else
  {
    game_time += delta;
    game_delta = delta;
  }
}