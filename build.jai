
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

  set_working_directory(project_dir);

  options.output_executable_name = "penrose";
  options.output_path = "./build";
  options.text_output_flags = 0;
  options.backend = .X64;
  options.runtime_support_definitions = .OMIT;

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

  files := file_list("./");
  for files
  {
    if contains(it, "build.jai") continue;
    if !contains(it, ".jai") continue;
    add_build_file(it, workspace);
  }

  set_build_options_dc(.{do_output=false});
}

#run build();

#import "Basic";
#import "File_Utilities";
#import "String";
#import "Compiler";
#import "Bindings_Generator";