
raddbg_attached_data :: #run -> [] u8
{
	segment, ok := add_data_segment(".raddbg", .READ | .WRITE);
	if !ok
	{
		print("RADDBG section will not be created, sorry!\n");
		return .[0];
	}

	attached_data := add_global_data(.[0], .USER_SEGMENT, segment);

	add_global_data(xx "entry_point: \"main\"\0", .USER_SEGMENT, segment);

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

	add_view_rule(segment, "? (?)", "disasm($)");
	add_view_rule(segment, "[]??", "rows($, count, slice($))");
	add_view_rule(segment, "[..]??", "rows($, count, slice($))");
	add_view_rule(segment, "string", "text($.data)");
	add_view_rule(segment, V2_f32, "rows($, x, y)");
	add_view_rule(segment, V3_f32, "rows($, x, y, z)");
	add_view_rule(segment, V4_f32, "rows($, x, y, z, w)");
	add_view_rule(segment, V2_f64, "rows($, x, y)");
	add_view_rule(segment, V3_f64, "rows($, x, y, z)");
	add_view_rule(segment, V4_f64, "rows($, x, y, z, w)");

	return attached_data;
}

raddbg_attached_data_pointer :: #run raddbg_attached_data.data;

raddbg_is_attached :: () -> bool
{
	return raddbg_attached_data_pointer.* == 1;
}

raw :: (values: ..Any) #expand {}
bin :: (values: ..Any) #expand {}
oct :: (values: ..Any) #expand {}
dec :: (values: ..Any) #expand {}
hex :: (values: ..Any) #expand {}
digits :: (values: ..Any) #expand {}
no_string :: (values: ..Any) #expand {}
no_char :: (values: ..Any) #expand {}
no_addr :: (values: ..Any) #expand {}
sequence :: (values: ..Any) #expand {}
rows :: (values: ..Any) #expand {}
omit :: (values: ..Any) #expand {}
array :: (values: ..Any) #expand {}
slice :: (values: ..Any) #expand {}
columns :: (values: ..Any) #expand {}
text :: (values: ..Any) #expand {}
disasm :: (values: ..Any) #expand {}
memory :: (values: ..Any) #expand {}
bitmap :: (values: ..Any) #expand {}
color :: (values: ..Any) #expand {}
range1 :: (values: ..Any) #expand {}
raddbg_pin :: (values: ..Any) #expand {}

#scope_file

#import "Compiler";