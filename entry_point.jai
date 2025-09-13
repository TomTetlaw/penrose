
SDL_WINDOW_VULKAN :: 0x0000000010000000;
SDL_WINDOW_BORDERLESS :: 0x0000000000000010;
SDL_WINDOW_HIDDEN :: 0x0000000000000008;

SDL_WINDOWPOS_CENTERED_DISPLAY :: (x: u32) -> s32
{
  return cast(s32) (cast(u32) 0x2FFF0000 | x);
}



main :: ()
{
  SDL_Init(SDL_INIT_VIDEO);
  SDL_ShaderCross_Init();
  window_flags: u64 = SDL_WINDOW_VULKAN | SDL_WINDOW_BORDERLESS | SDL_WINDOW_HIDDEN;
  window := SDL_CreateWindow("Walker", 1, 1, window_flags);
  if window
  {
    display_count: s32;
    displays := SDL_GetDisplays(*display_count);
    display_mode := SDL_GetDesktopDisplayMode(displays[0]);
    window_pos := SDL_WINDOWPOS_CENTERED_DISPLAY(displays[0]);
    resolution := V2_f32.{xx display_mode.w, xx display_mode.h} * .75;
    SDL_SetWindowSize(window, xx resolution.x, xx resolution.y);
    SDL_SetWindowPosition(window, window_pos, window_pos);
    SDL_SetWindowRelativeMouseMode(window, true);
    device := SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, true, "vulkan");
    if device
    {
      SDL_ClaimWindowForGPUDevice(device, window);
      SDL_ShowWindow(window);
      sys_init(window, device, resolution);
      while main_loop := true
      {
        event: SDL_Event;
        while SDL_PollEvent(*event)
        {
          type := cast(SDL_EventType) event.type;
          if type ==
          {
            case .SDL_EVENT_QUIT; break main_loop;
            case .SDL_EVENT_KEY_DOWN;
            {
              if event.key.scancode == .SDL_SCANCODE_ESCAPE
              {
                break main_loop;
              }
            }
          }
        }
        sys_run_frame();
        arena_reset(base.frame_arena);
        reset_temporary_storage();
      }
    }
    else
    {
      fatal_error("Failed to create gpu device: %", sys_get_sdl_error());
    }
  }
  else
  {
    fatal_error("Failed to create window: %", sys_get_sdl_error());
  }
}