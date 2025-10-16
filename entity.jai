
Prop :: struct
{
  using #as base: Entity;
  base.mesh_path = "dr_default_cube_mesh";
  base.material_name = "dr_default_material";
}

PlayerStart :: struct
{
  using #as base: Entity;
  base.mesh_path = "dr_default_cube_mesh";
  base.material_name = "info_player_start";
  base.name = "player_start";
  base.editor_only = true;
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

UsableCustomId :: enum
{
  Test;
}

UsableObject :: struct
{
  using #as base: Entity;
  base.collision.enabled = false;
  pixel_size := Vec2F32.{64, 64};
  type: UsableType;
  custom_id: UsableCustomId;
  target_name: string;
}

DynamicLight :: struct
{
  using #as base: Entity;
  base.collision.enabled = false;
  light_info: DR_LightInfo;
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
  base.material_name = "dr_default_material";
  base.scale = Vec3F32.{0.5, 3.0, 7.0};
  open: bool;
}

MeshTest :: struct
{
  using #as base: Entity;
  base.mesh_path = "mesh_export";
}