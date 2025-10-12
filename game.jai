
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
  bounds_min: Vec3F32;
  bounds_max: Vec3F32;
}

CollisionCapsule :: struct
{
  center: Vec3F32;
  top: Vec3F32;
  bottom: Vec3F32;
  radius: F32;
  height: F32;
}

g_collision_capsule :: (entity: *Entity) -> CollisionCapsule
{
  bottom := entity.position - v3f32(0, 0, entity.collision.capsule_height*.5);
  top := entity.position + v3f32(0, 0, entity.collision.capsule_height*.5);
  result: CollisionCapsule;
  result.center = entity.position;
  result.top = top;
  result.bottom = bottom;
  result.radius = entity.collision.capsule_radius;
  result.height = entity.collision.capsule_height;
  return result;
}

g_collision_box :: (entity: *Entity) -> CollisionBox
{
  if entity.mesh
  {
    local_min := entity.mesh.bounds_min * entity.scale * entity.collision.box_scale;
    local_max := entity.mesh.bounds_max * entity.scale * entity.collision.box_scale;
    bounds_min := entity.position + local_min;
    bounds_max := entity.position + local_max;
    return .{bounds_min, bounds_max};
  }
  else
  {
    local_min := v3f32(-.5) * entity.scale * entity.collision.box_scale;
    local_max := v3f32(.5) * entity.scale * entity.collision.box_scale;
    bounds_min := entity.position + local_min;
    bounds_max := entity.position + local_max;
    return .{bounds_min, bounds_max};
  }
}

CollisionResult :: struct
{
  hit: B8;
  normal: Vec3F32;
  depth: F32;
}

collision_check_box_capsule :: (box: CollisionBox, capsule: CollisionCapsule) -> CollisionResult
{
  box_closest := clamp_v3(capsule.center, box.bounds_min, box.bounds_max);
  ab := capsule.top - capsule.bottom;
  t := dot_v3(box_closest - capsule.bottom, ab) / dot_v3(ab, ab);
  t = clamp(t, 0.0, 1.0);
  segment_closest := capsule.bottom + ab * t;
  delta := box_closest - segment_closest;
  d := dot_v3(delta, delta);
  if d <= capsule.radius * capsule.radius
  {
    normal := normalize_v3(delta);
    return .{true, normal, capsule.radius - sqrt(d)};
  }
  return .{};
}

collision_check :: (a: *Entity, b: *Entity) -> CollisionResult
{
  if a.collision.type == .Box && b.collision.type == .Capsule
  {
    capsule := g_collision_capsule(b);
    box := g_collision_box(a);
    result := collision_check_box_capsule(box, capsule);
    return result;
  }
  if a.collision.type == .Capsule && b.collision.type == .Box
  {
    capsule := g_collision_capsule(a);
    box := g_collision_box(b);
    result := collision_check_box_capsule(box, capsule);
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
  material_name: string; @no_serialize @no_imgui
  name: string;
  mesh: *DR_Mesh;  @no_imgui
  spawn_position: Vec3F32; @no_serialize @no_imgui
  position: Vec3F32;
  rotation: Vec3F32;
  scale := Vec3F32.{1, 1, 1};
  material: *DR_MeshMaterial;
  colour := Vec4F32.{1, 1, 1, 1}; @colour
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
  entity.material = dr_get_material("dr_default_material");
  if entity.material_name
  {
    entity.material = dr_get_material(entity.material_name);
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
  g_game_state.player.position = player_start.position;
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
        colour := v4f32(1,0,0,1);
        if collision.hit
        {
          colour = v4f32(0,1,0,1);
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
      box := g_collision_box(door);
      span := box.bounds_max - box.bounds_min;
      open_z := door.spawn_position.z + span.z;
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
      dr_draw_mesh(it.mesh, .[instance], it.material);
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