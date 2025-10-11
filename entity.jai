
Prop :: struct
{
  using #as base: Entity;
  base.mesh_path = "dr_default_cube_mesh";
  base.texture_paths[0] = "textures/test_col.png";
  base.texture_paths[1] = "textures/test_norm.png";
  base.texture_paths[2] = "textures/test_spec.png";
}

PlayerStart :: struct
{
  using #as base: Entity;
  base.mesh_path = "dr_default_cube_mesh";
  base.texture_paths[0] = "textures/info_player_start.png";
  base.name = "player_start";
  base.editor_only = true;
  base.material.lighting_intensity = 0.0;
}

Player :: struct
{
  using #as base: Entity;
  base.collision.type = .Capsule;
  base.collision.capsule_height = 5.0;
  base.collision.capsule_radius = 1.0;
  move_speed := 6.0;
  camera: CAM_Player; @no_serialize  @no_imgui
}

UsableType :: enum
{
  Custom;
  TriggerDoor;
}

UsableObject :: struct
{
  using #as base: Entity;
  base.collision.enabled = false;
  pixel_size := Vec2F32.{64, 64};
  type: UsableType;
  custom_id: s32;
  target_name: string;
}

usable_in_focus :: (usable: *UsableObject) -> bool
{
  if g_game_state.mode == .Game
  {
    player := g_game_state.player;
    view := player.camera.view;
    proj := player.camera.projection;
    usable_clip := proj * view * v4f32(usable.position, 1.0);
    usable_screen := ((usable_clip.xy / usable_clip.w) * .5 + .5) * dr_renderer.resolution;
    usable_screen_min := usable_screen - usable.pixel_size*.5 - dr_renderer.resolution*.5;
    usable_screen_max := usable_screen + usable.pixel_size*.5 - dr_renderer.resolution*.5;
    if usable_screen_min.x < -usable.pixel_size.x return false;
    if usable_screen_min.y < -usable.pixel_size.y return false;
    if usable_screen_max.x >  usable.pixel_size.x return false;
    if usable_screen_max.y >  usable.pixel_size.y return false;
    return true;
  }
  return false;
}

Door :: struct
{
  using #as base: Entity;
  base.mesh_path = "dr_default_cube_mesh";
  base.texture_paths[0] = "textures/shipping_crates3.tga";
  base.texture_paths[1] = "textures/shipping_crates3_local.tga";
  base.texture_paths[2] = "textures/shipping_crates3_s.tga";
  base.scale = Vec3F32.{0.5, 3.0, 7.0};
  open: bool;
}