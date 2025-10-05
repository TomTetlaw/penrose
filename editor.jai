
ED_Prop :: struct
{
  next: *ED_Prop;
  prev: *ED_Prop;
  mesh: *DR_Mesh;
  position: Vec3F32;
  rotation: Vec3F32;
  scale: Vec3F32;
  material: DR_MeshMaterial;
  colour: Vec4F32;
}

ED_TransformMode :: enum
{
  Translate;
  Scale;
  Rotate;
}

ED_EditorState :: struct
{
  arena: *Arena;
  camera: CAM_Flying;
  props: *ED_Prop;
  last_prop: *ED_Prop;
  selected_prop: *ED_Prop;
  dragging_axis: bool;
  dragging_axis_index: int;
  drag_start_pos: Vec3F32;
  drag_start_object_pos: Vec3F32;
  drag_start_object_scale: Vec3F32;
  drag_start_object_rotation: Vec3F32;
  transform_mode: ED_TransformMode;
}

ed_editor: *ED_EditorState;

ed_create_prop :: (mesh: *DR_Mesh, position: Vec3F32) -> ED_Prop
{
  prop := arena_push_struct(ed_editor.arena, ED_Prop);
  list_add(*ed_editor.props, *ed_editor.last_prop, prop);
  prop.position = position;
  snap_v3(*prop.position, 0.5);
  prop.rotation = v3f32(0.0);
  prop.scale = v3f32(1.0);
  prop.colour = v4f32(1.0);
  prop.mesh = dr_renderer.mesh_cube;
  prop.material.tex_colour = tex_load("textures/shipping_crates3.tga");
  prop.material.tex_normal = tex_load("textures/shipping_crates3_local.tga");
  prop.material.tex_specular = tex_load("textures/shipping_crates3_s.tga");
  prop.material.specular_shininess = 32;
  return prop;
}

ed_init :: ()
{
  arena := mem_create_arena();
  ed_editor = arena_push_struct(arena, ED_EditorState);
  ed_editor.arena = arena;
}

ed_update :: ()
{
  cam_update(*ed_editor.camera);
  if in_key_pressed(.Space)
  {
    ed_create_prop(dr_renderer.mesh_cube, ed_editor.camera.position + ed_editor.camera.forward*10);
  }
  if ed_editor.selected_prop
  {
    prop := ed_editor.selected_prop;
    bounds_min := prop.mesh.bounds_min;
    bounds_max := prop.mesh.bounds_max;
    rotation := mat4x4f32_rotate(v3f32(1, 0, 0), prop.rotation.x) *
      mat4x4f32_rotate(v3f32(0, 1, 0), prop.rotation.y) *
      mat4x4f32_rotate(v3f32(0, 0, 1), prop.rotation.z);
    if in_key_pressed(.Tab)
    {
      enum_cycle(*ed_editor.transform_mode);
    }
    if ed_editor.transform_mode ==
    {
      case .Translate;
        update_translate_widget(prop.position, rotation, prop.scale, bounds_min, bounds_max, *prop.position);
      case .Scale;
        update_scale_widget(prop.position, prop.rotation, prop.scale, bounds_min, bounds_max, *prop.scale);
      case .Rotate;
        update_rotate_widget(prop.position, prop.rotation, prop.scale, bounds_min, bounds_max, *prop.rotation);
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
    for each_element(ed_editor.props)
    {
      transform := mat4x4f32_transform(it.position, it.rotation, it.scale);
      bounds_min := transform * v4f32(it.mesh.bounds_min, 1.0);
      bounds_max := transform * v4f32(it.mesh.bounds_max, 1.0);
      hit, t := intersection_rayf32_boundsf32(ray, bounds_min.xyz, bounds_max.xyz);
      if hit
      {
        ed_editor.selected_prop = it;
        return true;
      }
    }
    ed_editor.selected_prop = null;
    return true;
  }
  return false;
}

update_translate_widget :: (position: Vec3F32, rotation: Mat4x4F32, scale: Vec3F32, bounds_min: Vec3F32, bounds_max: Vec3F32, dest: *Vec3F32)
{
  axis_length :: 5.0;
  line_bounds_size :: .25;
  axis_dirs := Vec3F32.[
    v3f32(1, 0, 0),
    v3f32(0, 1, 0),
    v3f32(0, 0, 1),
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
  min_t := f32_max;
  world_pos := position * scale;
  for * lines
  {
    axis_dir := v4f32(axis_dirs[it_index], 0);
    transformed_axis := rotation * axis_dir;
    it.start = world_pos;
    it.end = world_pos + normalize_v3(transformed_axis.xyz) * axis_length;
    it.colour = v4f32(axis_dirs[it_index], 1.0);
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
    ed_editor.drag_start_object_pos = position;
  }
  if drag_state.ended
  {
    ed_editor.dragging_axis = false;
  }
  if ed_editor.dragging_axis
  {
    axis_dir := axis_dirs[ed_editor.dragging_axis_index];
    dr_draw_lines(.[.{start=world_pos - axis_dir*100000, end=world_pos + axis_dir*100000, colour=v4f32(axis_dir, 1), props=line_props(1)}]);
    closest := closest_point_on_line(ray.origin, ray.dir, ed_editor.drag_start_pos, axis_dir);
    delta := dot_v3(closest - ed_editor.drag_start_pos, axis_dir);
    result := ed_editor.drag_start_object_pos + axis_dir*delta/scale[ed_editor.dragging_axis_index];
    if in_input_state.alt_down
    {
      snap_v3(*result, 0.5);
    }
    dest.* = result;
  }
}

get_facing_bounds :: (bounds_min: Vec3F32, bounds_max: Vec3F32, to_world: Mat4x4F32) -> [3] Vec3F32, [3] Vec3F32
{
  forward := ed_editor.camera.forward;
  result_min: [3] Vec3F32;
  result_max: [3] Vec3F32;
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
  for 0..2
  {
    result_min[it] = (to_world * v4f32(result_min[it], 1.0)).xyz;
    result_max[it] = (to_world * v4f32(result_max[it], 1.0)).xyz;
  }
  return result_min, result_max;
}

update_scale_widget :: (position: Vec3F32, rotation: Vec3F32, scale: Vec3F32, bounds_min: Vec3F32, bounds_max: Vec3F32, dest: *Vec3F32)
{
  transform := mat4x4f32_transform(position, rotation, scale);
  axis_length :: 5.0;
  bounds_size :: .25;
  axis_dirs := Vec3F32.[
    v3f32(1, 0, 0),
    v3f32(0, 1, 0),
    v3f32(0, 0, 1),
  ];
  scale_bounds_min, scale_bounds_max := get_facing_bounds(bounds_min, bounds_max, transform);
  ray := rayf32_from_mouse(
    in_input_state.mouse_position,
    dr_renderer.resolution,
    ed_editor.camera.view,
    ed_editor.camera.projection
  );
  axis := -1;
  hit_pos: Vec3F32;
  min_t := f32_max;
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
    ed_editor.drag_start_object_scale = scale;
  }
  if drag_state.ended
  {
    ed_editor.dragging_axis = false;
  }
  if ed_editor.dragging_axis
  {
    axis := axis_dirs[ed_editor.dragging_axis_index];
    dr_draw_lines(.[.{start=position - axis*100000, end=position + axis*100000, colour=v4f32(axis, 1), props=line_props(1)}]);
    closest := closest_point_on_line(ray.origin, ray.dir, ed_editor.drag_start_pos, axis);
    delta := dot_v3(closest - ed_editor.drag_start_pos, axis);
    result := ed_editor.drag_start_object_scale;
    result[ed_editor.dragging_axis_index] *= max(1.0, (1.0 + delta));
    if in_input_state.alt_down
    {
      snap_v3(*result, .5);
    }
    dest.* = result;
  }
}

update_rotate_widget :: (position: Vec3F32, rotation: Vec3F32, scale: Vec3F32, bounds_min: Vec3F32, bounds_max: Vec3F32, dest: *Vec3F32)
{
  transform := mat4x4f32_transform(position, rotation, scale);
  axis_length :: 5.0;
  bounds_size :: .25;
  axis_dirs := Vec3F32.[
    v3f32(1, 0, 0),
    v3f32(0, 1, 0),
    v3f32(0, 0, 1),
  ];
  scale_bounds_min, scale_bounds_max := get_facing_bounds(bounds_min, bounds_max, transform);
  ray := rayf32_from_mouse(
    in_input_state.mouse_position,
    dr_renderer.resolution,
    ed_editor.camera.view,
    ed_editor.camera.projection
  );
  axis := -1;
  hit_pos: Vec3F32;
  min_t := f32_max;
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
    mid := (scale_bounds_min[it] + scale_bounds_max[it]) * .5;
    span := length_v3(scale_bounds_max[it] - scale_bounds_min[it]) * .5;
    dr_draw_circle(mid, axis_dirs[it], span, 64, colour, 1.0);
  }
  drag_state := in_mouse_drag(ed_editor.dragging_axis, .Left);
  if drag_state.started && axis != -1
  {
    ed_editor.dragging_axis = true;
    ed_editor.dragging_axis_index = axis;
    ed_editor.drag_start_pos = hit_pos;
    ed_editor.drag_start_object_rotation = rotation;
  }
  if drag_state.ended
  {
    ed_editor.dragging_axis = false;
  }
  if ed_editor.dragging_axis
  {
    axis := axis_dirs[ed_editor.dragging_axis_index];
    dr_draw_lines(.[.{start=position - axis*100000, end=position + axis*100000, colour=v4f32(axis, 1), props=line_props(1)}]);
    closest := closest_point_on_line(ray.origin, ray.dir, ed_editor.drag_start_pos, axis);
    delta := dot_v3(closest - ed_editor.drag_start_pos, axis);
    result := ed_editor.drag_start_object_rotation + axis*delta*.5;
    dest.* = result;
  }
}

ed_draw :: ()
{
  for each_element(ed_editor.props)
  {
    instance: DR_MeshInstance;
    instance.transform = mat4x4f32_transform(it.position, it.rotation, it.scale);
    instance.colour = it.colour;
    dr_draw_mesh(it.mesh, .[instance], *it.material);
    if ed_editor.selected_prop == it && !ed_editor.dragging_axis
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