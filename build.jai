
build :: ()
{
  project_dir := get_working_directory();
  workspace := compiler_create_workspace("Penrose build");
  release_build := false;
  options := get_build_options();
  args := options.compile_time_command_line;
  if array_find(args, "release")
  {
    release_build = true;
  }
  options.output_executable_name = "penrose";
  options.output_path = "build";
  options.text_output_flags = 0;
  options.backend = .LLVM;
  options.additional_linker_arguments = .["user32.lib", "imm32.lib", "shell32.lib"];
  if (release_build)
  {
    set_optimization(*options, .VERY_OPTIMIZED);
  }
  else
  {
    options.debug_for_expansions = true;
    set_optimization(*options, .VERY_DEBUG);
  }
  set_build_options(options, workspace);
  make_directory_if_it_does_not_exist("./build/shaders");
  compiler_begin_intercept(workspace);
  files := file_list("./");
  for files
  {
    if contains(it, ".jai") && !contains(it, "build.jai") && !contains(it, "_generated.jai")
    {
      add_build_file(it, workspace);
    }
    if contains(it, ".hlsl")
    {
      path, base, ext := path_decomp(it);
      copy_file(tprint("%.hlsl", base), tprint("build/shaders/%.hlsl", base));
    }
  }
  texture_files := file_list("build/textures");
  for texture_files
  {
    if !contains(it, "_spec") && !contains(it, "_norm")
    {
      path, base, ext := path_decomp(it);
      col := tprint("textures/%.%", base, ext);
      norm := tprint("textures/%_norm.%", base, ext);
      spec := tprint("textures/%_spec.%", base, ext);
      if !file_exists(tprint("build/%", norm))
      {
        norm = "dr_default_normal_texture";
      }
      if !file_exists(tprint("build/%", spec))
      {
        spec = "dr_default_white_texture";
      }
      if !file_exists(tprint("build/materials/%.txt", base))
      {
        builder: String_Builder;
        print_to_builder(*builder, "name: %\n", base);
        print_to_builder(*builder, "tex_colour: %\n", col);
        print_to_builder(*builder, "tex_normal: %\n", norm);
        print_to_builder(*builder, "tex_specular: %\n", spec);
        mat := builder_to_string(*builder);
        write_entire_file(tprint("build/materials/%.txt", base), mat);
      }
    }
  }
  entity_types: [..] *Type_Info_Struct;
  generated_code := false;
  while true
  {
    message := compiler_wait_for_message();
    if message.kind ==
    {
      case .COMPLETE;
        break;
      case .TYPECHECKED;
        typechecked := cast(*Message_Typechecked) message;
        for typechecked.structs
        {
          if is_subclass_of(it.expression.defined_type, "Entity")
          {
            array_add_if_unique(*entity_types, it.expression.defined_type);
          }
          if it.expression.defined_type.name == "Entity"
          {
            array_add_if_unique(*entity_types, it.expression.defined_type);
          }
        }
      case .PHASE;
        phase := cast(*Message_Phase) message;
        if phase.phase == .TYPECHECKED_ALL_WE_CAN
        {
          if !generated_code
          {
            generated_code = true;
            code := generate_code(entity_types);
            write_entire_file("entity_generated.jai", code);
            add_build_string(code, workspace);
          }
        }
    }
  }
  compiler_end_intercept(workspace);
  set_build_options_dc(.{do_output=false});
}

CreateEntityFromStringCode :: #string HERE
g_create_entity_from_type_name :: (str: string) -> *Entity #expand
{
  if str ==
  {
%1
case; return g_create_entity(Entity);
  }
}
HERE

EntityTypeInfoCode :: #string HERE
g_entity_type_info := EntityTypeInfo.[
%1
];
HERE

SerializeEntityCode :: #string HERE
g_write_entity :: (base_entity: *Entity, file: *File)
{
  if base_entity.info.type ==
  {
    %1
  }
}
HERE

DeserializeEntityCode :: #string HERE
g_read_entity_member :: (base_entity: *Entity, key: string, value: string)
{
  if base_entity.info.type ==
  {
    %1
  }
}
HERE

EntityImGUICode :: #string HERE
g_entity_imgui :: (base_entity: *Entity)
{
  if base_entity.info.type ==
  {
    %1
  }
}
HERE

PerMemberCallback :: #type (info: *Type_Info, name: string, base_name: string) -> string;

generate_imgui_member_code :: (info: *Type_Info, name: string, base_name: string) -> string
{
  if info.type ==
  {
    case .POINTER;
      pointer_to := cast(*Type_Info_Pointer) info;
      return generate_imgui_member_code(pointer_to.pointer_to, name, base_name);
    case .ARRAY;
      element_type := cast(*Type_Info_Array) info;
      return generate_imgui_member_code(element_type.element_type, name, base_name);
    case .STRUCT;
    {
      struct_info := cast(*Type_Info_Struct) info;
      if struct_info.name ==
      {
        case "V2";
          return tprint("vec2_property(\"%\", *%);\n", name, name);
        case "V3";
          return tprint("vec3_property(\"%\", *%);\n", name, name);
        case "V4";
          if array_find(struct_info.notes, "colour")
          {
            return tprint("colour_property(\"%\", *%);\n", name, name);
          }
          else
          {
            return tprint("vec4_property(\"%\", *%);\n", name, name);
          }
        case "DR_MeshMaterial";
          return tprint("material_property(\"%\", *%);\n", name, name);
      }
    }
    case .INTEGER;
      return tprint("ImGui.DragInt(\"%\", *%);\n", name, name);
    case .FLOAT;
      return tprint("ImGui.DragFloat(\"%\", *%);\n", name, name);
    case .STRING;
      return tprint("string_property(\"%\", *%);\n", name, name);
    case .BOOL;
      return tprint("ImGui.Checkbox(\"%\", *%);\n", name, name);
    case .ENUM;
      return tprint("enum_property(\"%\", *%);\n", name, name);
  }
  return "";
}

generate_deserialize_member_code :: (info: *Type_Info, name: string, base_name: string) -> string
{
  if info.type ==
  {
    case .POINTER;
      pointer_to := cast(*Type_Info_Pointer) info;
      return generate_deserialize_member_code(pointer_to.pointer_to, name, base_name);
    case .ARRAY;
      element_type := cast(*Type_Info_Array) info;
      return generate_deserialize_member_code(element_type.element_type, name, base_name);
    case .STRUCT;
    {
      struct_info := cast(*Type_Info_Struct) info;
      if struct_info.name ==
      {
        case "V2";
          return tprint("if key == \"%1\" %1 = parse_v2(value);\n", name);
        case "V3";
          return tprint("if key == \"%1\" %1 = parse_v3(value);\n", name);
        case "V4";
          return tprint("if key == \"%1\" %1 = parse_v4(value);\n", name);
        case "DR_Mesh";
          return tprint("if key == \"%1\" %1 = dr_get_mesh(value);\n", name);
        case "DR_MeshMaterial";
          return tprint("if key == \"%1\" %1 = dr_get_material(value);\n", name);
      }
    }
    case .INTEGER;
      return tprint("if key == \"%1\" %1 = xx parse_int(*value);\n", name);
    case .FLOAT;
      return tprint("if key == \"%1\" %1 = parse_float(*value);\n", name);
    case .STRING;
      return tprint("if key == \"%1\" %1 = value;\n", name);
    case .BOOL;
      return tprint("if key == \"%1\" %1 = parse_bool(*value);\n", name);
    case .ENUM;
      return tprint("if key == \"%1\" %1 = parse_enum(*value, type_of(%1));\n", name);
  }
  return name;
}

generate_serialize_member_code :: (info: *Type_Info, name: string, base_name: string) -> string
{
  if info.type ==
  {
    case .POINTER;
      pointer_to := cast(*Type_Info_Pointer) info;
      return generate_serialize_member_code(pointer_to.pointer_to, name, base_name);
    case .ARRAY;
      element_type := cast(*Type_Info_Array) info;
      return generate_serialize_member_code(element_type.element_type, name, base_name);
    case .STRUCT;
    {
      struct_info := cast(*Type_Info_Struct) info;
      if struct_info.name ==
      {
        case "V2";
          return tprint("file_print_v2(file, \"%1\", %1);\n", name);
        case "V3";
          return tprint("file_print_v3(file, \"%1\", %1);\n", name);
        case "V4";
          return tprint("file_print_v4(file, \"%1\", %1);\n", name);
        case "DR_Mesh";
          return tprint("if %1 file_print_line(file, \"%1: \%\", %1.path);\n", name);
        case "DR_MeshMaterial";
          return tprint("if %1 file_print_line(file, \"%1: \%\", %1.name);\n", name);
      }
    }
    case;
      return tprint("file_print_line(file, \"%1: \%\", %1);\n", name);
  }
  return name;
}

generate_per_member_code :: (info: *Type_Info_Struct, base: string, callback: PerMemberCallback, note_filter: string) -> string
{
  filtered_members :: string.["next", "prev"];
  handled_types :: string.["DR_Mesh", "DR_Material", "V2", "V3", "V4"];
  builder: String_Builder;
  for info.members
  {
    if !array_find(filtered_members, it.name) && !array_find(it.notes, note_filter)
    {
      if it.type.type == .STRUCT
      {
        info := cast(*Type_Info_Struct) it.type;
        if array_find(handled_types, info.name)
        {
          code := callback(it.type, tprint("%.%", base, it.name), it.name);
          append(*builder, code);
        }
        else
        {
          code := generate_per_member_code(info, tprint("%.%", base, it.name), callback, note_filter);
          append(*builder, code);
        }
      }
      else
      {
        code := callback(it.type, tprint("%.%", base, it.name), it.name);
        append(*builder, code);
      }
    }
  }
  return builder_to_string(*builder);
}

generate_code :: (entity_types: [] *Type_Info_Struct) -> string
{
  full_builder: String_Builder;
  {
    builder: String_Builder;
    for entity_types
    {
      code := generate_per_member_code(it, "entity", generate_serialize_member_code, "no_serialize");
      print_to_builder(*builder, "case %1; entity := g_downcast(base_entity, %1);\n%\n", it.name, code);
    }
    code := tprint(SerializeEntityCode, builder_to_string(*builder));
    append(*full_builder, code);
  }
  {
    builder: String_Builder;
    for entity_types
    {
      code := generate_per_member_code(it, "entity", generate_deserialize_member_code, "no_serialize");
      print_to_builder(*builder, "case %1; entity := g_downcast(base_entity, %1);\n%\n", it.name, code);
    }
    code := tprint(DeserializeEntityCode, builder_to_string(*builder));
    append(*full_builder, code);
  }
  {
    builder: String_Builder;
    for entity_types
    {
      code := generate_per_member_code(it, "entity", generate_imgui_member_code, "no_imgui");
      print_to_builder(*builder, "case %1; entity := g_downcast(base_entity, %1);\n%\n", it.name, code);
    }
    code := tprint(EntityImGUICode, builder_to_string(*builder));
    append(*full_builder, code);
  }
  {
    builder: String_Builder;
    for entity_types
    {
      print_to_builder(*builder, "    case \"%1\"; return g_create_entity(%1);\n", it.name);
    }
    code := tprint(CreateEntityFromStringCode, builder_to_string(*builder));
    append(*full_builder, code);
  }
  {
    builder: String_Builder;
    for entity_types
    {
      print_to_builder(*builder, "  .{\"%1\", %1, size_of(%1)},\n", it.name);
    }
    code := tprint(EntityTypeInfoCode, builder_to_string(*builder));
    append(*full_builder, code);
  }
  return builder_to_string(*full_builder);
}

#run build();

#scope_file

#import "Basic";
#import "File";
#import "File_Utilities";
#import "String";
#import "Compiler";
#import "Bindings_Generator";
#import "Process";
#import "BuildCpp";