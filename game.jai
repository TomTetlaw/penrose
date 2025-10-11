
G_GameMode :: enum
{
  Editor;
  Game;
}

G_GameState :: struct
{
  arena: *Arena;
  mode: G_GameMode;
  entities_arena: *Arena;
  entities: *Entity;
  last_entity: *Entity;
  player: *Player;
}

CollisionType :: enum
{
  None;
  Box;
  Capsule;
}

CollisionInfo :: struct
{
  type: CollisionType;
  enabled := true;
  box_scale := Vec3F32.{1, 1, 1};
  capsule_radius: F32;
  capsule_height: F32;
}

CollisionBox :: struct
{
  box_min: Vec3F32;
  box_max: Vec3F32;
}

CollisionCapsule :: struct
{
  capsule_center: Vec3F32;
  capsule_radius: F32;
  capsule_height: F32;
}

CollisionResult :: struct
{
  hit: B8;
  normal: Vec3F32;
  depth: F32;
}

collision_check_box_capsule :: (a: CollisionBox, b: CollisionCapsule) -> CollisionResult
{
  capsule_bottom := b.capsule_center - v3f32(0, 0, b.capsule_height*.5);
  capsule_top := b.capsule_center + v3f32(0, 0, b.capsule_height*.5);
  box_closest := clamp_v3(b.capsule_center, a.box_min, a.box_max);
  ab := capsule_top - capsule_bottom;
  t := dot_v3(box_closest - capsule_bottom, ab) / dot_v3(ab, ab);
  t = clamp(t, 0.0, 1.0);
  segment_closest := capsule_bottom + ab * t;
  delta := box_closest - segment_closest;
  d := dot_v2(delta.xy, delta.xy);
  if d <= b.capsule_radius * b.capsule_radius
  {
    normal := normalize_v3(delta);
    return .{true, normal, b.capsule_radius - sqrt(d)};
  }
  return .{};
}

collision_check :: (a: *Entity, b: *Entity) -> CollisionResult
{
  if a.collision.type == .Capsule && b.collision.type == .Box
  {
    box_min_b, box_max_b := g_entity_world_bounds(b, b.collision.box_scale);
    capsule_center := a.position;
    result := collision_check_box_capsule(.{box_min_b, box_max_b}, .{capsule_center, a.collision.capsule_radius, a.collision.capsule_height});
    result.normal = -result.normal;
    return result;
  }
  return .{};
}

EntityTypeInfo :: struct
{
  type_name: string;
  type: Type;
  size: int;
}

Entity :: struct
{
  next: *Entity; @no_serialize @no_imgui
  prev: *Entity; @no_serialize @no_imgui
  info: EntityTypeInfo; @no_serialize @no_imgui
  editor_only: bool; @no_serialize @no_imgui
  removed: bool; @no_serialize @no_imgui
  mesh_path := ""; @no_serialize @no_imgui
  texture_paths: [3] string; @no_serialize @no_imgui
  name: string;
  mesh: *DR_Mesh;  @no_imgui
  spawn_position: Vec3F32; @no_serialize @no_imgui
  position: Vec3F32;
  rotation: Vec3F32;
  scale := Vec3F32.{1, 1, 1};
  material: DR_MeshMaterial;
  colour := Vec4F32.{1, 1, 1, 1};
  collision: CollisionInfo;
  collision.type = .Box;
}

g_game_state: *G_GameState;

g_init :: ()
{
  arena := mem_create_arena();
  g_game_state = arena_push_struct(arena, G_GameState);
  g_game_state.arena = arena;
  g_game_state.entities_arena = mem_create_arena();
}

g_create_entity :: ($T: Type) -> *T
{
  entity := cast(*Entity) arena_push_struct(g_game_state.entities_arena, T);
  list_add(*g_game_state.entities, *g_game_state.last_entity, entity);
  struct_info := cast(*Type_Info_Struct) type_info(T);
  for g_entity_type_info
  {
    if it.type_name == struct_info.name
    {
      entity.info = it;
      break;
    }
  }
  if entity.mesh_path
  {
    entity.mesh = dr_get_mesh(entity.mesh_path);
  }
  entity.material.tex_colour = dr_renderer.tex_white;
  entity.material.tex_normal = dr_renderer.tex_normal;
  entity.material.tex_specular = dr_renderer.tex_white;
  if entity.texture_paths[0]
  {
    entity.material.tex_colour = dr_get_texture(entity.texture_paths[0]);
  }
  if entity.texture_paths[1]
  {
    entity.material.tex_normal = dr_get_texture(entity.texture_paths[1]);
  }
  if entity.texture_paths[2]
  {
    entity.material.tex_specular = dr_get_texture(entity.texture_paths[2]);
  }
  return g_downcast(entity, T);
}

g_downcast :: (entity: *Entity, $T: Type) -> *T
{
  if entity.info.type != T
  {
    return null;
  }
  return cast(*T) entity;
}

g_find_entity_by_name :: (name: string, filter := EntityFilter.{}) -> *Entity
{
  for each_entity(filter)
  {
    if it.name == name
    {
      return it;
    }
  }
  return null;
}

g_clear_level :: ()
{
  arena_reset(g_game_state.entities_arena);
  g_game_state.entities = null;
  g_game_state.last_entity = null;
}

g_start_level :: ()
{
  set_fps_mode(true);
  player_start := g_find_entity_by_name("player_start", .{include_editor_only = true});
  g_game_state.player = g_create_entity(Player);
  g_game_state.player.position = v3f32(player_start.position.xy, player_start.position.z);
  g_game_state.mode = .Game;
  for each_entity()
  {
    it.spawn_position = it.position;
  }
}

g_remove_entity :: (entity: *Entity)
{
  entity.removed = true;
}

g_update :: ()
{
  if g_game_state.mode == .Game
  {
    player := g_game_state.player;
    movement: Vec3F32;
    movement.x = in_key_state(.W) - in_key_state(.S);
    movement.y = in_key_state(.A) - in_key_state(.D);
    movement = rotate_v3(mat4x4f32_rotate(v3f32(0, 0, 1), player.camera.yaw), movement);
    player.position += v3f32(movement.xy, 0.0) * player.move_speed * game_delta;
    for each_entity(.{exclude_types = .[Player]})
    {
      if it.collision.enabled
      {
        collision := collision_check(player, it);
        if collision.hit
        {
          resolve := player.position + collision.normal * collision.depth;
          player.position.x = resolve.x;
          player.position.y = resolve.y;
        }
      }
    }
    player.camera.position = v3f32(player.position.xy, player.position.z + player.collision.capsule_height*.5);
    for each_entity(.{include_types = .[Door]})
    {
      door := g_downcast(it, Door);
      open_z := door.spawn_position.z + 8.0;
      close_z := door.spawn_position.z;
      if door.open
      {
        door.position.z = approach(door.position.z, open_z, 1.0, game_delta);
      }
      else
      {
        door.position.z = approach(door.position.z, close_z, 1.0, game_delta);
      }
    }
    cam_player_update(*player.camera);
    if in_key_pressed(.Escape)
    {
      g_clear_level();
      game_paused = true;
      g_game_state.mode = .Editor;
      ed_load(ed_editor.level_name);
      set_fps_mode(false);
    }
    if in_key_pressed(.E)
    {
      for each_entity(.{include_types = .[UsableObject]})
      {
        usable := g_downcast(it, UsableObject);
        if usable_in_focus(usable)
        {
          handle_usable(usable);
        }
      }
    }
  }
}

handle_usable :: (usable: *UsableObject)
{
  if usable.type ==
  {
    case .Custom;
      if usable.custom_id ==
      {
        case 1;
          print("Hello!\n");
      }
    case .TriggerDoor;
      door := g_downcast(g_find_entity_by_name(usable.target_name), Door);
      if door
      {
        door.open = !door.open;
        door.collision.enabled = !door.open;
      }
  }
}

g_draw :: (filter := EntityFilter.{})
{
  for each_entity(filter)
  {
    if it.mesh
    {
      instance: DR_MeshInstance;
      instance.transform = mat4x4f32_transform(it.position, it.rotation, it.scale);
      instance.colour = it.colour;
      dr_draw_mesh(it.mesh, .[instance], *it.material);
    }
  }
  for each_entity(.{include_types = .[UsableObject]})
  {
    usable := g_downcast(it, UsableObject);
    quad: DR_QuadInstance;
    quad.position = it.position;
    if usable_in_focus(usable)
    {
      quad.colour = v4f32(0, 1, 0, 1);
    }
    else
    {
      quad.colour = v4f32(212.0, 188.0, 125.0, 255.0) / 255.0;
    }
    quad.props = quad_props(usable.pixel_size, 1.0);
    dr_draw_quads(.[quad]);
  }
}

g_entity_world_bounds :: (entity: *Entity, extra_scale := Vec3F32.{1, 1, 1}) -> Vec3F32, Vec3F32
{
  if entity.mesh
  {
    bounds_min := entity.position + entity.mesh.bounds_min*entity.scale*extra_scale;
    bounds_max := entity.position + entity.mesh.bounds_max*entity.scale*extra_scale;
    return bounds_min, bounds_max;
  }
  return .{}, .{};
}

EntityFilter :: struct
{
  include_types: [] Type;
  exclude_types: [] Type;
  include_editor_only: bool;
}

each_entity :: (filter := EntityFilter.{}) -> EntityFilter
{
  return filter;
}

is_entity_filtered :: (entity: *Entity, filter: EntityFilter) -> bool
{
  filtered := entity.removed ||
    (!filter.include_editor_only && entity.editor_only) ||
    (filter.include_types.count > 0 && !array_find(filter.include_types, entity.info.type)) ||
    (filter.exclude_types.count > 0 && array_find(filter.exclude_types, entity.info.type));
  return filtered;
}

for_expansion :: (filter: EntityFilter, body: Code, flags: For_Flags) #expand
{
  for `it, `it_index: each_element(g_game_state.entities)
  {
    if !is_entity_filtered(it, filter)
    {
      #insert body;
    }
  }
}

g_save_level :: (path: string, filter := EntityFilter.{})
{
  file := file_open(path, for_writing = true);
  defer file_close(*file);
  for each_entity(filter)
  {
    file_print_line(*file, "entity: %", it.info.type);
    g_write_entity(it, *file);
  }
}

g_load_level :: (path: string)
{
  g_clear_level();
  file_data, ok := read_entire_file(path,, arena_allocator(arena_temp));
  if ok
  {
    entity: *Entity;
    _, lines := file_to_array("", true, true, false, true, file_data);
    for lines
    {
      key, right := break_by_spaces(it);
      key.count -= 1; // NOTE(tt): skip the ':'
      if key == "entity"
      {
        entity = g_create_entity_from_type_name(right);
      }
      else
      {
        g_read_entity_member(entity, key, right);
      }
    }
  }
}

file_print_time :: (file: *File, name: string, t: F32)
{
  file_write(file, str_format(arena_temp, "%: %\n", t - game_time));
}

file_print_v2 :: (file: *File, name: string, v: Vec2F32)
{
  file_write(file, str_format(arena_temp, "%: [%, %]\n", name, v.x, v.y));
}

file_print_v3 :: (file: *File, name: string, v: Vec3F32)
{
  file_write(file, str_format(arena_temp, "%: [%, %, %]\n", name, v.x, v.y, v.z));
}

file_print_v4 :: (file: *File, name: string, v: Vec4F32)
{
  file_write(file, str_format(arena_temp, "%: [%, %, %, %]\n", name, v.x, v.y, v.z, v.w));
}

parse_v2 :: (str: string) -> Vec2F32
{
  x, y: F32;
  scan2(str, "[%, %]", *x, *y);
  result: Vec2F32;
  result.x = x;
  result.y = y;
  return result;
}

parse_v3 :: (str: string) -> Vec3F32
{
  x, y, z: F32;
  scan2(str, "[%, %, %]", *x, *y, *z);
  result: Vec3F32;
  result.x = x;
  result.y = y;
  result.z = z;
  return result;
}

parse_v4 :: (str: string) -> Vec4F32
{
  x, y, z, w: F32;
  scan2(str, "[%, %, %, %]", *x, *y, *z, *w);
  result: Vec4F32;
  result.x = x;
  result.y = y;
  result.z = z;
  result.w = w;
  return result;
}

parse_time :: (str: string) -> F32
{
  return game_time + parse_float(*str);
}