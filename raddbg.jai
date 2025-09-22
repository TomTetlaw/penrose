
raddbg_attached_data :: #run -> [] u8
{
	add_view_rule :: (segment: *Data_Segment, type_string: string, rule: string)
	{
		rule_string := tprint("type_view: {type: ```%```, expr: ```%```}\0", type_string, rule);
		add_global_data(xx rule_string, .USER_SEGMENT, segment);
	}
	add_view_rule :: (segment: *Data_Segment, $T: Type, rule: string)
	{
		struct_info := cast (*Type_Info_Struct) type_info(T);
		rule_string := tprint("type_view: {type: ```%```, expr: ```%```}\0", struct_info.name, rule);
		replace_chars(rule_string, "\n\t", #char " ");
		add_global_data(xx rule_string, .USER_SEGMENT, segment);
	}

	segment, ok := add_data_segment(".raddbg", .READ | .WRITE);
	if !ok
	{
		print("RADDBG section will not be created, sorry!\n");
		return .[0];
	}
	attached_data := add_global_data(.[0], .USER_SEGMENT, segment);
	add_global_data(xx "entry_point: \"entry_point\"\0", .USER_SEGMENT, segment);
	add_view_rule(segment, "? (?)", "disasm($)");
	add_view_rule(segment, "[]??", "rows($, count, slice($))");
	add_view_rule(segment, "[..]??", "rows($, count, slice($))");
	return attached_data;
}

raddbg_attached_data_pointer :: #run raddbg_attached_data.data;
raddbg_is_attached :: () -> bool
{
	return raddbg_attached_data_pointer.* == 1;
}

raddbg_annotate_vaddr_range :: (pointer: *void, size: u64, label: string)
{
  if raddbg_is_attached()
  {
    RADDBG_VaddrRangeAnnotationInfo :: struct
    {
      vaddr: u64;
      size: u64;
      name: *void;
      name_size: u64;
    }
    info: RADDBG_VaddrRangeAnnotationInfo;
    info.vaddr = cast(u64) pointer;
    info.size = size;
    info.name = label.data;
    info.name_size = xx label.count;
    RaiseException(0x00524156, 0, 4, *info);
  }
}

#scope_file

kernel32 :: #library,system "kernel32";
RaiseException :: (code: u32, flags: u32, arg_count: u32, args: *void) #foreign kernel32;

using,only
(
  Data_Segment,
  add_data_segment,
  add_global_data
) Compiler :: #import "Compiler";

using,only
(
  tprint
) Basic :: #import "Basic";