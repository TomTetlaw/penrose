
ED_TransformMode :: enum
{
  Translate;
  Scale;
}

ED_EditorState :: struct
{
  arena: *Arena;
  camera: CAM_Flying;
  selected_entity: *Entity;
  dragging_axis: bool;
  dragging_axis_index: int;
  drag_start_pos: Vec3F32;
  drag_start_object_pos: Vec3F32;
  drag_start_object_scale: Vec3F32;
  drag_start_object_rotation: Vec3F32;
  transform_mode: ED_TransformMode;
}

ed_editor: *ED_EditorState;

ed_spawn :: (mesh: *DR_Mesh, position: Vec3F32) -> *Entity
{
  entity := g_spawn(mesh, position);
  snap_v3(*entity.position, 0.5);
  return entity;
}

ed_init :: ()
{
  arena := mem_create_arena();
  ed_editor = arena_push_struct(arena, ED_EditorState);
  ed_editor.arena = arena;
}

ED_TransformInfo :: struct
{
  bounds_min: Vec3F32;
  bounds_max: Vec3F32;
  position: Vec3F32;
  rotation: Vec3F32;
  scale: Vec3F32;
  mat_translate: Mat4x4F32;
  mat_rotate: Mat4x4F32;
  mat_scale: Mat4x4F32;
  full: Mat4x4F32;
}

ed_update :: ()
{
  cam_update(*ed_editor.camera);
  if in_key_pressed(.F5)
  {
    g_save_level("levels/test_level.txt");
  }
  if in_key_pressed(.Space)
  {
    ed_spawn(dr_renderer.mesh_cube, ed_editor.camera.position + ed_editor.camera.forward*10);
  }
  if ed_editor.selected_entity
  {
    if in_key_pressed(.Tab)
    {
      enum_cycle(*ed_editor.transform_mode);
    }
    entity := ed_editor.selected_entity;
    bounds_min := entity.mesh.bounds_min;
    bounds_max := entity.mesh.bounds_max;
    rotation := mat4x4f32_rotate(v3f32(1, 0, 0), entity.rotation.x) *
      mat4x4f32_rotate(v3f32(0, 1, 0), entity.rotation.y) *
      mat4x4f32_rotate(v3f32(0, 0, 1), entity.rotation.z);
    transform: ED_TransformInfo;
    transform.bounds_min = bounds_min;
    transform.bounds_max = bounds_max;
    transform.position = entity.position;
    transform.rotation = entity.rotation;
    transform.scale = entity.scale;
    transform.mat_translate = mat4x4f32_translate(entity.position);
    transform.mat_rotate = rotation;
    transform.mat_scale = mat4x4f32_scale(entity.scale);
    transform.full = transform.mat_translate * transform.mat_rotate * transform.mat_scale;
    if ed_editor.transform_mode ==
    {
      case .Translate;
        update_translate_widget(transform, *entity.position);
      case .Scale;
        update_scale_widget(transform, *entity.scale);
    }
  }
  update_selection();
}

update_selection :: () -> B8
{
  if ed_editor.dragging_axis
  {
    return false;
  }
  pressed, pos := in_mouse_pressed(.Left);
  if pressed
  {
    ray := rayf32_from_mouse(
      pos,
      dr_renderer.resolution,
      ed_editor.camera.view,
      ed_editor.camera.projection
    );
    selected: *Entity;
    min_t := MaxF32;
    for each_element(g_game_state.entities)
    {
      transform := mat4x4f32_transform(it.position, it.rotation, it.scale);
      bounds_min := transform * v4f32(it.mesh.bounds_min, 1.0);
      bounds_max := transform * v4f32(it.mesh.bounds_max, 1.0);
      hit, t := intersection_rayf32_boundsf32(ray, bounds_min.xyz, bounds_max.xyz);
      if hit && t < min_t
      {
        selected = it;
        min_t = min(min_t, t);
      }
    }
    if selected
    {
      ed_editor.selected_entity = selected;
      return true;
    }
    else
    {
      ed_editor.selected_entity = null;
      return false;
    }
  }
  return false;
}

update_translate_widget :: (transform: ED_TransformInfo, dest: *Vec3F32)
{
  axis_length :: 5.0;
  line_bounds_size :: .25;
  axis_dirs := Vec3F32.[
    rotate_v3(v3f32(1, 0, 0), transform.rotation),
    rotate_v3(v3f32(0, 1, 0), transform.rotation),
    rotate_v3(v3f32(0, 0, 1), transform.rotation)
  ];
  lines: [3] DR_LineInstance;
  line_bounds_min: [3] Vec3F32;
  line_bounds_max: [3] Vec3F32;
  hit: [3] B8;
  hit_pos: [3] Vec3F32;
  ray := rayf32_from_mouse(
    in_input_state.mouse_position,
    dr_renderer.resolution,
    ed_editor.camera.view,
    ed_editor.camera.projection
  );
  axis := -1;
  min_t := MaxF32;
  for * lines
  {
    axis_dir := v4f32(axis_dirs[it_index], 0);
    it.start = transform.position;
    it.end = transform.position + normalize_v3(axis_dir.xyz) * axis_length;
    it.colour = v4f32(0, 0, 0, 1);
    it.colour[it_index] = 1.0;
    it.props = line_props(5.0);
    line_bounds_min[it_index] = it.start - (v3f32(1.0) - axis_dirs[it_index])*line_bounds_size;
    line_bounds_max[it_index] = it.end   + (v3f32(1.0) - axis_dirs[it_index])*line_bounds_size;
    axis_hit, t := intersection_rayf32_boundsf32(ray, line_bounds_min[it_index], line_bounds_max[it_index]);
    if axis_hit && t < min_t
    {
      axis = it_index;
      min_t = t;
    }
    hit[it_index] = axis_hit;
    hit_pos[it_index] = ray.origin + t*ray.dir;
  }
  if axis >= 0
  {
    lines[axis].colour = v4f32(1.0);
  }
  dr_draw_lines(lines);
  drag_state := in_mouse_drag(ed_editor.dragging_axis, .Left);
  if drag_state.started && axis != -1
  {
    ed_editor.dragging_axis = true;
    ed_editor.dragging_axis_index = axis;
    ed_editor.drag_start_pos = hit_pos[axis];
    ed_editor.drag_start_object_pos = transform.position;
  }
  if drag_state.ended
  {
    ed_editor.dragging_axis = false;
  }
  if ed_editor.dragging_axis
  {
    axis_dir := axis_dirs[ed_editor.dragging_axis_index];
    dr_draw_lines(.[.{
      start = transform.position - axis_dir*100000,
      end = transform.position + axis_dir*100000,
      colour = v4f32(axis_dir, 1),
      props = line_props(1)
    }]);
    closest := closest_point_on_line(ray.origin, ray.dir, ed_editor.drag_start_pos, axis_dir);
    delta := dot_v3(closest - ed_editor.drag_start_pos, axis_dir);
    result := ed_editor.drag_start_object_pos + axis_dir*delta;
    if !in_input_state.alt_down
    {
      snap_v3(*result, 0.5);
    }
    dest.* = result;
  }
}

update_scale_widget :: (transform: ED_TransformInfo, dest: *Vec3F32)
{
  axis_length :: 5.0;
  bounds_size :: .25;
  axis_dirs := Vec3F32.[
    rotate_v3(v3f32(1, 0, 0), transform.rotation),
    rotate_v3(v3f32(0, 1, 0), transform.rotation),
    rotate_v3(v3f32(0, 0, 1), transform.rotation)
  ];
  scale_bounds_min, scale_bounds_max := get_facing_bounds(transform);
  ray := rayf32_from_mouse(
    in_input_state.mouse_position,
    dr_renderer.resolution,
    ed_editor.camera.view,
    ed_editor.camera.projection
  );
  axis := -1;
  hit_pos: Vec3F32;
  min_t := MaxF32;
  for 0..2
  {
    axis_hit, t := intersection_rayf32_boundsf32(ray, scale_bounds_min[it], scale_bounds_max[it]);
    if axis_hit && t < min_t
    {
      axis = it;
      min_t = t;
      hit_pos = ray.origin + t*ray.dir;
    }
  }
  for 0..2
  {
    colour := v4f32(0.0, 0.0, 0.0, 1.0);
    if axis == it
    {
      colour = v4f32(1.0);
    }
    else
    {
      colour[it] = 1.0;
    }
    dr_draw_bounds(scale_bounds_min[it], scale_bounds_max[it], mat4x4f32_identity(), colour, 1.0, .02);
  }
  drag_state := in_mouse_drag(ed_editor.dragging_axis, .Left);
  if drag_state.started && axis != -1
  {
    ed_editor.dragging_axis = true;
    ed_editor.dragging_axis_index = axis;
    ed_editor.drag_start_pos = hit_pos;
    ed_editor.drag_start_object_scale = transform.scale;
  }
  if drag_state.ended
  {
    ed_editor.dragging_axis = false;
  }
  if ed_editor.dragging_axis
  {
    axis := axis_dirs[ed_editor.dragging_axis_index];
    dr_draw_lines(.[.{
      start = transform.position - axis*100000,
      end = transform.position + axis*100000,
      colour = v4f32(axis, 1),
      props = line_props(1)
    }]);
    closest := closest_point_on_line(ray.origin, ray.dir, ed_editor.drag_start_pos, axis);
    drag_vec := closest - transform.position;
    proj_len := dot_v3(drag_vec, axis);
    start_len := dot_v3(ed_editor.drag_start_pos - transform.position, axis);
    factor := proj_len / start_len;
    result := ed_editor.drag_start_object_scale;
    result[ed_editor.dragging_axis_index] *= factor;
    result[ed_editor.dragging_axis_index] = max(.01, result[ed_editor.dragging_axis_index]);
    if !in_input_state.alt_down
    {
        snap_v3(*result, 0.5);
    }
    dest.* = result;
  }
}

ed_draw :: ()
{
  for each_element(g_game_state.entities)
  {
    instance: DR_MeshInstance;
    instance.transform = mat4x4f32_transform(it.position, it.rotation, it.scale);
    instance.colour = it.colour;
    dr_draw_mesh(it.mesh, .[instance], *it.material);
    if ed_editor.selected_entity == it && !ed_editor.dragging_axis
    {
      colour := lerp_v4(v4f32(0.0, 1.0, 0.0, 1.0), v4f32(1.0, 1.0, 1.0, .25), sin(game_time*5)*.5+.5);
      dr_draw_bounds(it.mesh.bounds_min, it.mesh.bounds_max, instance.transform, colour, 1.0, .01);
    }
  }
}

#scope_file

closest_point_on_line :: (ray_origin: Vec3F32, ray_dir: Vec3F32, line_point: Vec3F32, line_dir: Vec3F32) -> Vec3F32
{
  d1 := normalize_v3(ray_dir);
  d2 := normalize_v3(line_dir);
  r := ray_origin - line_point;
  a := dot_v3(d1, d1);
  b := dot_v3(d1, d2);
  c := dot_v3(d2, d2);
  d := dot_v3(d1, r);
  e := dot_v3(d2, r);
  denom := a*c - b*b;
  t := 0.0;
  if abs(denom) > .000001
  {
    t = (b*e - c*d) / denom;
  }
  else
  {
    t = 0.0;
  }
  closest_point := ray_origin + d1 * t;
  return closest_point;
}

get_facing_bounds :: (transform: ED_TransformInfo) -> [3] Vec3F32, [3] Vec3F32
{
  bounds_min := (transform.full * v4f32(transform.bounds_min, 1.0)).xyz;
  bounds_max := (transform.full * v4f32(transform.bounds_max, 1.0)).xyz;
  result_min: [3] Vec3F32;
  result_max: [3] Vec3F32;
  forward := ed_editor.camera.forward;
  if forward.x < 0
  {
    result_min[0] = v3f32(bounds_max.x, bounds_min.y, bounds_min.z);
    result_max[0] = v3f32(bounds_max.x, bounds_max.y, bounds_max.z);
  }
  else
  {
    result_min[0] = v3f32(bounds_min.x, bounds_min.y, bounds_min.z);
    result_max[0] = v3f32(bounds_min.x, bounds_max.y, bounds_max.z);
  }
  if forward.y < 0
  {
    result_min[1] = v3f32(bounds_min.x, bounds_max.y, bounds_min.z);
    result_max[1] = v3f32(bounds_max.x, bounds_max.y, bounds_max.z);
  }
  else
  {
    result_min[1] = v3f32(bounds_min.x, bounds_min.y, bounds_min.z);
    result_max[1] = v3f32(bounds_max.x, bounds_min.y, bounds_max.z);
  }
  if forward.z < 0
  {
    result_min[2] = v3f32(bounds_min.x, bounds_min.y, bounds_max.z);
    result_max[2] = v3f32(bounds_max.x, bounds_max.y, bounds_max.z);
  }
  else
  {
    result_min[2] = v3f32(bounds_min.x, bounds_min.y, bounds_min.z);
    result_max[2] = v3f32(bounds_max.x, bounds_max.y, bounds_min.z);
  }
  return result_min, result_max;
}