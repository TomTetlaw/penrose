
sys_init :: (window: *SDL_Window, device: *SDL_GPUDevice, resolution: Vec2F32)
{
  base_init();
  dr_init(window, device, resolution);
}

sys_run_frame :: ()
{
  dr_draw_frame();
}

sys_get_sdl_error :: () -> string
{
  return to_string(SDL_GetError());
}