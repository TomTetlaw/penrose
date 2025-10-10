
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
  base.name = "player_start";
  base.texture_paths[0] = "textures/info_player_start.png";
  base.editor_only = true;
  base.material.lighting_intensity = 0.0;
}

Player :: struct
{
  using #as base: Entity;
  base.collision.type = .Capsule;
  base.collision.capsule_height = 6.0;
  base.collision.capsule_radius = 1.0;
  move_speed := 6.0;
  camera: CAM_Player;
}