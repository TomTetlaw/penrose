
sys_init :: (window: *SDL_Window, device: *SDL_GPUDevice, resolution: V2)
{
  
}

sys_run_frame :: ()
{

}

sys_get_sdl_error :: () -> string
{
  return to_string(SDL_GetError());
}