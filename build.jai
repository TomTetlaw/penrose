
build :: ()
{
  project_dir := get_working_directory();
  workspace := compiler_create_workspace("Penrose build");
  release_build := false;
  profiling := false;
  options := get_build_options();
  args := options.compile_time_command_line;
  if array_find(args, "release")
  {
    release_build = true;
  }
  if array_find(args, "profile")
  {
    profiling = true;
  }
  options.output_executable_name = "penrose";
  options.output_path = "build";
  options.text_output_flags = 0;
  options.backend = .LLVM;
  options.llvm_options.output_llvm_ir = true;
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
  plugins: [..] *Metaprogram_Plugin;
  to_create: [..] Plugin_To_Create;
  if profiling
  {
    array_add(*to_create, .{name="tracy"});
  }
  init_plugins(to_create, *plugins, workspace);
  intercept_flags: Intercept_Flags;
  for plugins if it.before_intercept it.before_intercept(it, *intercept_flags);
  compiler_begin_intercept(workspace);
  add_build_string(tprint("PENROSE_RELEASE :: %;\n", release_build), workspace);
  add_build_string(tprint("PENROSE_DEBUG :: %;\n", !release_build), workspace);
  add_build_string(tprint("PENROSE_PROFILE :: %;\n", profiling), workspace);
  for plugins if it.add_source it.add_source(it);
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
  while true
  {
    message := compiler_wait_for_message();
    for plugins if it.message it.message(it, message);
    if message.kind ==
    {
      case .COMPLETE;
        break;
    }
  }
  compiler_end_intercept(workspace);
  for plugins if it.finish it.finish(it);
  for plugins if it.shutdown it.shutdown(it);
  set_build_options_dc(.{do_output=false});
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
#import "Metaprogram_Plugins";