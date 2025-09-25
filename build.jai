
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
  files := file_list("./");
  for files
  {
    if contains(it, ".jai") && !contains(it, "build.jai")
    {
      add_build_file(it, workspace);
    }
    if contains(it, ".hlsl")
    {
      path, base, ext := path_decomp(it);
      copy_file(tprint("%.hlsl", base), tprint("build/shaders/%.hlsl", base));
    }
  }
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