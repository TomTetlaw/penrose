//
// This file was auto-generated using the following command:
//
// jai generate.jai
//



STB_DXT_NORMAL :: 0;
STB_DXT_DITHER :: 1;
STB_DXT_HIGHQUAL :: 2;

stb_compress_dxt_block :: (dest: *u8, src_rgba_four_bytes_per_pixel: *u8, alpha: s32, mode: s32) -> void #foreign stb_dxt;
stb_compress_bc4_block :: (dest: *u8, src_r_one_byte_per_pixel: *u8) -> void #foreign stb_dxt;
stb_compress_bc5_block :: (dest: *u8, src_rg_two_byte_per_pixel: *u8) -> void #foreign stb_dxt;

#scope_file

#import "Basic"; // For assert

stb_dxt :: #library,no_dll "lib/stb_dxt";
