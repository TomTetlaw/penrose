
G_GameState :: struct
{
  arena: *Arena;
  entities_arena: *Arena;
  entities: *Entity;
  last_entity: *Entity;
}

CollisionType :: enum
{
  CollisionType_Box;
  CollisionType_Capsule;
}

CollisionInfo :: struct
{
  type: CollisionType;
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
  colliding: B8;
  normal: Vec3F32;
}

collision_check_box_box :: (a: CollisionBox, b: CollisionBox) -> CollisionResult
{
  if (a.box_max.x < b.box_min.x || a.box_min.x > b.box_max.x) return .{};
  if (a.box_max.y < b.box_min.y || a.box_min.y > b.box_max.y) return .{};
  if (a.box_max.z < b.box_min.z || a.box_min.z > b.box_max.z) return .{};
  overlap := min_v3(a.box_max, b.box_max) - max_v3(a.box_min, b.box_min);
  min_overlap := overlap.x;
  normal := v3f32(1, 0, 0);
  if overlap.y < min_overlap
  {
    min_overlap = overlap.y;
    normal = v3f32(0, 1, 0);
  }
  if overlap.z < min_overlap
  {
    min_overlap = overlap.z;
    normal = v3f32(0,0,1);
  }
  center_a := (a.box_min + a.box_max)*.5;
  center_b := (b.box_min + b.box_max)*.5;
  delta := center_b - center_a;
  if dot_v3(delta, normal) < 0.0 normal = -normal;
  return .{true, normal};
}

closest_point_on_segment :: (p: Vec3F32, capsule_top: Vec3F32, capsule_bottom: Vec3F32) -> Vec3F32
{
  ab := capsule_top - capsule_bottom;
  t := dot_v3(p - capsule_bottom, ab) / dot_v3(ab, ab);
  t = clamp(t, 0.0, 1.0);
  return capsule_bottom + ab * t;
}

collision_check_box_capsule :: (a: CollisionBox, b: CollisionCapsule) -> CollisionResult
{
  capsule_bottom := b.capsule_center - b.capsule_height*.5 - b.capsule_radius*.5;
  capsule_top := b.capsule_center + b.capsule_height*.5 + b.capsule_radius*.5;
  box_closest := clamp_v3(b.capsule_center, a.box_min, a.box_max);
  segment_closest := closest_point_on_segment(box_closest, capsule_top, capsule_bottom);
  delta := box_closest - segment_closest;
  dist_sq := dot_v3(delta, delta);
  if dist_sq <= b.capsule_radius * b.capsule_radius
  {
    dist := sqrt(dist_sq);
    normal := normalize_v3(delta);
    return .{true, normal};
  }
  return .{};
}

g_entity_world_bounds :: (entity: *Entity, extra_scale := Vec3F32.{1, 1, 1}) -> Vec3F32, Vec3F32
{
  bounds_min := entity.position + entity.mesh.bounds_min*entity.scale*extra_scale;
  bounds_max := entity.position + entity.mesh.bounds_max*entity.scale*extra_scale;
  return bounds_min, bounds_max;
}

collision_check :: (a: *Entity, b: *Entity) -> CollisionResult
{
  if a.collision.type == .CollisionType_Box && b.collision.type == .CollisionType_Box
  {
    box_min_a, box_max_a := g_entity_world_bounds(a, a.collision.box_scale);
    box_min_b, box_max_b := g_entity_world_bounds(b, b.collision.box_scale);
    return collision_check_box_box(.{box_min_a, box_max_a}, .{box_min_b, box_max_b});
  }
  if a.collision.type == .CollisionType_Box && b.collision.type == .CollisionType_Capsule
  {
    box_min_a, box_max_a := g_entity_world_bounds(a, a.collision.box_scale);
    capsule_center := b.position;
    return collision_check_box_capsule(.{box_min_a, box_max_a}, .{capsule_center, b.collision.capsule_height, b.collision.capsule_radius});
  }
  if a.collision.type == .CollisionType_Capsule && b.collision.type == .CollisionType_Box
  {
    box_min_b, box_max_b := g_entity_world_bounds(b, b.collision.box_scale);
    capsule_center := a.position;
    result := collision_check_box_capsule(.{box_min_b, box_max_b}, .{capsule_center, a.collision.capsule_height, a.collision.capsule_radius});
    result.normal = -result.normal;
    return result;
  }
  return .{};
}

Entity :: struct
{
  next: *Entity;
  prev: *Entity;
  mesh: *DR_Mesh;
  position: Vec3F32;
  rotation: Vec3F32;
  scale: Vec3F32;
  material: DR_MeshMaterial;
  colour: Vec4F32;
  collision: CollisionInfo;
}

g_game_state: *G_GameState;

g_init :: ()
{
  arena := mem_create_arena();
  g_game_state = arena_push_struct(arena, G_GameState);
  g_game_state.arena = arena;
  g_game_state.entities_arena = mem_create_arena();
}

g_create_entity :: () -> *Entity
{
  entity := arena_push_struct(g_game_state.entities_arena, Entity);
  list_add(*g_game_state.entities, *g_game_state.last_entity, entity);
  return entity;
}

g_spawn :: (mesh: *DR_Mesh, position: Vec3F32) -> *Entity
{
  entity := g_create_entity();
  entity.position = position;
  entity.rotation = v3f32(0.0);
  entity.scale = v3f32(1.0);
  entity.colour = v4f32(1.0);
  entity.mesh = mesh;
  entity.collision.type = .CollisionType_Box;
  entity.collision.box_scale = v3f32(1.0);
  entity.material.tex_coord_scale = v2f32(1.0);
  entity.material.tex_colour = dr_get_texture("textures/shipping_crates3.tga");
  entity.material.tex_normal = dr_get_texture("textures/shipping_crates3_local.tga");
  entity.material.tex_specular = dr_get_texture("textures/shipping_crates3_s.tga");
  entity.material.specular_shininess = 32;
  return entity;
}

g_clear_level :: ()
{
  arena_reset(g_game_state.entities_arena);
  g_game_state.entities = null;
  g_game_state.last_entity = null;
}

g_save_level :: (path: string)
{
  file := file_open(path, for_writing = true);
  defer file_close(*file);
  for each_element(g_game_state.entities)
  {
    file_write_line(*file, "entity");
    file_print_v3(*file, "position", it.position);
    file_print_v3(*file, "rotation", it.rotation);
    file_print_v3(*file, "scale", it.scale);
    file_print_line(*file, "collision_type: %", enum_names(CollisionType)[it.collision.type]);
    file_print_v3(*file, "collision_box_scale", it.collision.box_scale);
    file_print_line(*file, "collision_capsule_radius: %", it.collision.capsule_radius);
    file_print_line(*file, "collision_capsule_height: %", it.collision.capsule_height);
    file_print_v4(*file, "colour", it.colour);
    file_print_line(*file, "mesh: %", it.mesh.path);
    file_print_line(*file, "material_tex_colour: %", it.material.tex_colour.path);
    file_print_line(*file, "material_tex_normal: %", it.material.tex_normal.path);
    file_print_line(*file, "material_tex_specular: %", it.material.tex_specular.path);
    file_print_line(*file, "material_specular_power: %", it.material.specular_power);
    file_print_line(*file, "material_specular_shininess: %", it.material.specular_shininess);
    file_print_line(*file, "material_specular_intensity: %", it.material.specular_intensity);
    file_print_v2(*file, "material_tex_coord_scale", it.material.tex_coord_scale);
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
      if key != "entity"
      {
        key.count -= 1; // NOTE(tt): skip the ':' for key/value lines
      }
      if key ==
      {
        case "entity";
          entity = g_create_entity();
        case "position";
          position := parse_v3(right);
          entity.position = position;
        case "rotation";
          rotation := parse_v3(right);
          entity.rotation = rotation;
        case "scale";
          scale := parse_v3(right);
          entity.scale = scale;
        case "collision_type";
          type := parse_enum(*right, CollisionType);
          entity.collision.type = type;
        case "collision_box_scale";
          box_scale := parse_v3(right);
          entity.collision.box_scale = box_scale;
        case "collision_capsule_radius";
          radius := parse_float(*right);
          entity.collision.capsule_radius = radius;
        case "collision_capsule_height";
          height := parse_float(*right);
          entity.collision.capsule_height = height;
        case "colour";
          colour := parse_v4(right);
          entity.colour = colour;
        case "mesh";
          mesh_path := right;
          entity.mesh = dr_get_mesh(mesh_path);
        case "material_tex_colour";
          tex_path := right;
          entity.material.tex_colour = dr_get_texture(tex_path);
        case "material_tex_normal";
          tex_path := right;
          entity.material.tex_normal = dr_get_texture(tex_path);
        case "material_tex_specular";
          tex_path := right;
          entity.material.tex_specular = dr_get_texture(tex_path);
        case "material_specular_shininess";
          shininess := parse_float(*right);
          entity.material.specular_shininess = shininess;
        case "material_specular_intensity";
          intensity := parse_float(*right);
          entity.material.specular_intensity = intensity;
        case "material_specular_power";
          power := parse_int(*right);
          entity.material.specular_power = xx power;
        case "material_tex_coord_scale";
          tex_coord_scale := parse_v2(right);
          entity.material.tex_coord_scale = tex_coord_scale;
      }
    }
  }
}

g_draw :: ()
{
  for each_element(g_game_state.entities)
  {
    instance: DR_MeshInstance;
    instance.transform = mat4x4f32_transform(it.position, it.rotation, it.scale);
    instance.colour = it.colour;
    dr_draw_mesh(it.mesh, .[instance], *it.material);
  }
}

#scope_file

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