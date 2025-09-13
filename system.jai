
sys_init :: (window: *SDL_Window, device: *SDL_GPUDevice, resolution: V2_f32)
{
  base_init();
  r_init(window, device, resolution);
}

sys_run_frame :: ()
{
  instance: R_QuadInstance;
  instance.position = V2_f32.{100, 100};
  instance.size = V2_f32.{100, 100};
  instance.colour = V4_f32.{1, 0, 0, 1};
  r_draw(.quads, R_QuadInstance.[instance]);
  r_draw_frame();
}

sys_get_sdl_error :: () -> string
{
  return to_string(SDL_GetError());
}