
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
            array_add(*entity_types, it.expression.defined_type);
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
    case "Entity"; #through;
    case; return g_create_entity(Entity);
%1
  }
}
HERE

EntityTypeInfoCode :: #string HERE
g_entity_type_info := EntityTypeInfo.[
  .{"Entity", Entity, size_of(Entity)},
%1
];
HERE

generate_code :: (entity_types: [] *Type_Info_Struct) -> string
{
  full_builder: String_Builder;
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