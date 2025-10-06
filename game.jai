
G_GameState :: struct
{
  arena: *Arena;
  entities: *Entity;
  last_entity: *Entity;
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
}

g_game_state: *G_GameState;

g_init :: ()
{
  arena := mem_create_arena();
  g_game_state = arena_push_struct(arena, G_GameState);
  g_game_state.arena = arena;
  g_load_level("levels/test_level.txt");
}

g_spawn :: (mesh: *DR_Mesh, position: Vec3F32) -> *Entity
{
  entity := arena_push_struct(g_game_state.arena, Entity);
  list_add(*g_game_state.entities, *g_game_state.last_entity, entity);
  entity.position = position;
  entity.rotation = v3f32(0.0);
  entity.scale = v3f32(1.0);
  entity.colour = v4f32(1.0);
  entity.mesh = mesh;
  entity.material.tex_colour = tex_load("textures/shipping_crates3.tga");
  entity.material.tex_normal = tex_load("textures/shipping_crates3_local.tga");
  entity.material.tex_specular = tex_load("textures/shipping_crates3_s.tga");
  entity.material.specular_shininess = 32;
  return entity;
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
    file_print_v4(*file, "colour", it.colour);
    file_print_line(*file, "mesh: %", it.mesh.path);
    file_print_line(*file, "material_tex_colour: %", it.material.tex_colour.path);
    file_print_line(*file, "material_tex_normal: %", it.material.tex_normal.path);
    file_print_line(*file, "material_tex_specular: %", it.material.tex_specular.path);
    #if PENROSE_USE_IMGUI
    {
      file_print_line(*file, "material_specular_power: %", it.material.specular_power);
    }
    file_print_line(*file, "material_specular_shininess: %", it.material.specular_shininess);
    file_print_line(*file, "material_specular_intensity: %", it.material.specular_intensity);
  }
}

g_load_level :: (path: string)
{
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
          entity = arena_push_struct(g_game_state.arena, Entity);
          list_add(*g_game_state.entities, *g_game_state.last_entity, entity);
        case "position";
          position := parse_v3(right);
          entity.position = position;
        case "rotation";
          rotation := parse_v3(right);
          entity.rotation = rotation;
        case "scale";
          scale := parse_v3(right);
          entity.scale = scale;
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
      }
      #if PENROSE_USE_IMGUI
      {
        if key ==
        {
          case "material_specular_power";
            power := parse_int(*right);
            entity.material.specular_power = xx power;
        }
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

texture_picker_modal :: (path: string, dest: **DR_Texture)
{
  if ImGui.BeginPopupModal(cstr_from_str(arena_temp, path))
  {
    texture_paths := file_list("textures");
    ImGui.BeginTable("textures", 3);
    for texture_paths
    {
      if it_index % 3 == 0
      {
        ImGui.TableNextRow();
      }
      ImGui.TableNextColumn();
      texture := dr_get_texture(it);
      if texture
      {
        ImGui.BeginGroup();
        ImGui.PushID(cast(s32) it_index);
        ImGui.Text(it);
        if ImGui.ImageButton("", .{_TexID = cast(ImTextureID) texture.ref}, ImVec2.{128, 128})
        {
          dest.* = texture;
          ImGui.CloseCurrentPopup();
        }
        ImGui.PopID();
        ImGui.EndGroup();
      }
    }
    ImGui.EndTable();
    ImGui.EndPopup();
  }
}

modal_path: string;
modal_dest: **DR_Texture;

g_entity_imgui :: (entity: *Entity)
{
  ImGui.Begin("Entity");
  ImGui.SeparatorText("transform");
  ImGui.DragFloat3("position", *entity.position.f, v_speed = .5);
  ImGui.DragFloat3("rotation", *entity.rotation.f, v_min = 0.0, v_max = 1.0, v_speed = .01);
  ImGui.DragFloat3("scale", *entity.scale.f, v_min = .5, v_speed = .5);
  ImGui.SeparatorText("material");
  ImGui.ColorEdit4("colour", *entity.colour.f);
  ImGui.DragFloat("specular_intensity", *entity.material.specular_intensity, v_min = 1.0, v_max = 10.0, v_speed = 0.1);
  ImGui.SliderInt("specular_shininess", *entity.material.specular_power, v_min = 2, v_max = 9);
  ImGui.SeparatorText("textures");
  if (ImGui.Button(cstr_from_str(arena_temp, str_format(arena_temp, "Colour texture: %", entity.material.tex_colour.path))))
  {
    modal_path = entity.material.tex_colour.path;
    modal_dest = *entity.material.tex_colour;
    ImGui.OpenPopup(cstr_from_str(arena_temp, modal_path));
  }
  if (ImGui.Button(cstr_from_str(arena_temp, str_format(arena_temp, "Normal texture: %", entity.material.tex_normal.path))))
  {
    modal_path = entity.material.tex_normal.path;
    modal_dest = *entity.material.tex_normal;
    ImGui.OpenPopup(cstr_from_str(arena_temp, modal_path));
  }
  if (ImGui.Button(cstr_from_str(arena_temp, str_format(arena_temp, "Specular texture: %", entity.material.tex_specular.path))))
  {
    modal_path = entity.material.tex_specular.path;
    modal_dest = *entity.material.tex_specular;
    ImGui.OpenPopup(cstr_from_str(arena_temp, modal_path));
  }

  texture_picker_modal(modal_path, modal_dest);

  entity.material.specular_shininess = cast(F32) 1 << entity.material.specular_power;
  ImGui.End();
}

#scope_file

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

file_print_v3 :: (file: *File, name: string, v: Vec3F32)
{
  file_write(file, str_format(arena_temp, "%: [%, %, %]\n", name, v.x, v.y, v.z));
}

file_print_v4 :: (file: *File, name: string, v: Vec4F32)
{
  file_write(file, str_format(arena_temp, "%: [%, %, %, %]\n", name, v.x, v.y, v.z, v.w));
}