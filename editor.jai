
ED_TransformMode :: enum
{
  Translate;
  Scale;
}

ED_EditorState :: struct
{
  arena: *Arena;
  level_name: string;
  camera: CAM_Editor;
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

ed_spawn :: (type_name: string, position: Vec3F32) -> *Entity
{
  entity := g_create_entity_from_type_name(type_name);
  entity.position = position;
  snap_v3(*entity.position, 0.5);
  return entity;
}

ed_save :: ()
{
  path := str_format(arena_temp, "levels/%.txt", ed_editor.level_name);
  g_save_level(path, .{include_editor_only = true});
}

ed_load :: (name: string)
{
  path := str_format(arena_temp, "levels/%.txt", name);
  g_load_level(path);
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
  cam_editor_update(*ed_editor.camera);
  if in_key_pressed(.Escape) || ImGui.IsKeyPressed(.Escape)
  {
    if ed_editor.selected_entity
    {
      ed_editor.selected_entity = null;
    }
  }
  if ed_editor.selected_entity
  {
    entity_imgui(ed_editor.selected_entity);
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
    if in_key_pressed(.Delete)
    {
      g_remove_entity(ed_editor.selected_entity);
      ed_editor.selected_entity = null;
    }
  }
  update_selection();
  editor_imgui();
  if in_key_pressed(.F5)
  {
    ed_save();
    g_start_level();
  }
}

ed_draw :: ()
{
  for each_entity(.{include_editor_only = true})
  {
    if ed_editor.selected_entity == it && !ed_editor.dragging_axis
    {
      colour := lerp_v4(v4f32(0.0, 1.0, 0.0, 1.0), v4f32(1.0, 1.0, 1.0, .25), sin(game_time*5)*.5+.5);
      if it.collision.type ==
      {
        case .CollisionType_Box;
        {
          box_min, box_max := g_entity_world_bounds(it, it.collision.box_scale);
          dr_draw_bounds(box_min, box_max, mat4x4f32_identity(), colour, 1.0, .01);
        }
        case .CollisionType_Capsule;
        {
          dr_draw_capsule(it.position, it.collision.capsule_radius, it.collision.capsule_height, 32, colour, 1.0);
        }
      }
    }
  }
}

#scope_file

editor_imgui :: ()
{
  if ed_editor.level_name
  {
    ImGui.Begin(cstr_from_str(arena_temp, str_format(arena_temp, "Level (%)###LevelWindow", ed_editor.level_name)));
  }
  else
  {
    ImGui.Begin("Level###LevelWindow");
  }
  if ImGui.Button("New")
  {
    g_clear_level();
  }
  if ImGui.Button("Load")
  {
    ImGui.OpenPopup("LevelPicker");
  }
  {
    if !ed_editor.level_name
    {
      ImGui.BeginDisabled();
    }
    if ImGui.Button("Save")
    {
      ed_save();
    }
    if !ed_editor.level_name
    {
      ImGui.EndDisabled();
    }
  }
  {
    buffer: [256] u8;
    copy(buffer.data, ed_editor.level_name.data, xx ed_editor.level_name.count);
    if ImGui.InputText("Name", buffer.data, 256)
    {
      ed_editor.level_name = str_from_cstr(ed_editor.arena, buffer.data);
    }
  }
  level_picker_modal();
  if ImGui.Button("Quit")
  {
    exit(0);
  }
  ImGui.SeparatorText("spawn");
  not_spawnable := string.["Player"];
  for g_entity_type_info
  {
    if !array_find(not_spawnable, it.type_name)
    {
      if ImGui.Button(cstr_from_str(arena_temp, it.type_name))
      {
        ed_spawn(it.type_name, ed_editor.camera.position + ed_editor.camera.forward*10);
      }
    }
  }
  ImGui.End();
  ImGui.Begin("Entities");
  for each_entity(.{include_editor_only = true})
  {
    if ImGui.Button(cstr_from_str(arena_temp, ifx it.name then it.name else it.info.type_name))
    {
      ed_editor.selected_entity = it;
    }
  }
  ImGui.End();
}

level_picker_modal :: ()
{
  if ImGui.BeginPopup("LevelPicker")
  {
    files := file_list("levels");
    for files
    {
      path, base, ext := path_decomp(it);
      if ImGui.Button(cstr_from_str(arena_temp, base))
      {
        ed_load(base);
        ImGui.CloseCurrentPopup();
      }
    }
    ImGui.EndPopup();
  }
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
    for each_entity(.{include_editor_only = true})
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

texture_picker_dest: **DR_Texture;

texture_picker_modal :: ()
{
  if texture_picker_dest && ImGui.BeginPopup("TexturePicker")
  {
    if ImGui.IsKeyPressed(.Escape)
    {
      ImGui.CloseCurrentPopup();
      texture_picker_dest = null;
    }
    paths: [..] string;
    paths.allocator = arena_allocator(arena_temp);
    texture_paths := file_list("textures");
    array_add(*paths, ..texture_paths);
    ImGui.BeginTable("textures", 3);
    for each_element(dr_renderer.textures)
    {
      if it_index % 3 == 0
      {
        ImGui.TableNextRow();
      }
      ImGui.TableNextColumn();
      texture := dr_get_texture(it.path);
      if texture
      {
        ImGui.PushID(cast(s32) it_index);
        ImGui.Text(it.path);
        if ImGui.ImageButton("", .{_TexID = cast(ImTextureID) texture.ref}, ImVec2.{64, 64})
        {
          texture_picker_dest.* = texture;
          texture_picker_dest = null;
          ImGui.CloseCurrentPopup();
        }
        ImGui.PopID();
      }
    }
    ImGui.EndTable();
    ImGui.EndPopup();
  }
}

texture_property :: (label: string, texture: **DR_Texture)
{
  path, base, ext := path_decomp(texture.*.path);
  label_str := str_format(arena_temp, "%: %", label, base);
  ImGui.Text(label);
  ImGui.SameLine();
  if (ImGui.Button(base.data))
  {
    texture_picker_dest = texture;
    ImGui.OpenPopup("TexturePicker");
  }
  ImGui.SameLine();
  ImGui.Image(.{_TexID = cast(ImTextureID) texture.*.ref}, .{32, 32});
}

entity_imgui :: (entity: *Entity)
{
  ImGui.Begin("Entity");
  draw_list := ImGui.GetWindowDrawList();
  draw_list.AddCallback(draw_list, set_sampler_callback, null);
  ImGui.SeparatorText("transform");
  ImGui.DragFloat3("position", *entity.position.f, v_speed = .5);
  ImGui.DragFloat3("rotation", *entity.rotation.f, v_min = 0.0, v_max = 1.0, v_speed = .01);
  ImGui.DragFloat3("scale", *entity.scale.f, v_min = .5, v_speed = .5);
  ImGui.SeparatorText("collision");
  if ImGui.BeginCombo("collision_type", cstr_from_str(arena_temp, enum_names(CollisionType)[entity.collision.type]))
  {
    for enum_values_as_enum(CollisionType)
    {
      if ImGui.Button(cstr_from_str(arena_temp, enum_names(CollisionType)[it]))
      {
        entity.collision.type = it;
      }
    }
    ImGui.EndCombo();
  }
  if entity.collision.type == .CollisionType_Box
  {
    ImGui.DragFloat3("box_scale", *entity.collision.box_scale.f);
  }
  if entity.collision.type == .CollisionType_Capsule
  {
    ImGui.DragFloat("capsule_radius", *entity.collision.capsule_radius);
    ImGui.DragFloat("capsule_height", *entity.collision.capsule_height);
  }
  ImGui.SeparatorText("material");
  ImGui.ColorEdit4("colour", *entity.colour.f);
  ImGui.DragFloat("specular_intensity", *entity.material.specular_intensity, v_min = 1.0, v_max = 10.0, v_speed = 0.1);
  if (ImGui.SliderInt("specular_power", *entity.material.specular_power, v_min = 2, v_max = 9))
  {
    entity.material.specular_shininess = cast(F32) 1 << entity.material.specular_power;
  }
  ImGui.DragFloat("lighting_intensity", *entity.material.lighting_intensity, v_min = 0.0, v_max = 1.0, v_speed = .01);
  ImGui.SeparatorText("textures");
  texture_property("colour", *entity.material.tex_colour);
  texture_property("normal", *entity.material.tex_normal);
  texture_property("specular", *entity.material.tex_specular);
  ImGui.DragFloat2("tex_coord_scale", *entity.material.tex_coord_scale.f);
  texture_picker_modal();
  ImGui.End();
}

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

set_sampler_callback :: (parent_list: *ImDrawList, command: *ImDrawCmd) #c_call
{
  push_context
  {
    state := cast(*ImGui_ImplSDLGPU3_RenderState) ImGui.GetPlatformIO().Renderer_RenderState;
    state.SamplerCurrent = dr_get_sampler(.LINEAR, .LINEAR, .REPEAT);
  }
}