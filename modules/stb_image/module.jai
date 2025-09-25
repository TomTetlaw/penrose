//
// This file was auto-generated using the following command:
//
// jai generate.jai
//



STBI_VERSION :: 1;

STBI :: enum s32 {
    default    :: 0;

    grey       :: 1;
    grey_alpha :: 2;
    rgb        :: 3;
    rgb_alpha  :: 4;

    STBI_default    :: default;

    STBI_grey       :: grey;
    STBI_grey_alpha :: grey_alpha;
    STBI_rgb        :: rgb;
    STBI_rgb_alpha  :: rgb_alpha;
}

stbi_uc :: u8;
stbi_us :: u16;

//
// load image by filename, open file, or memory buffer
//
stbi_io_callbacks :: struct {
    read: #type (user: *void, data: *u8, size: s32) -> s32 #c_call; // fill 'data' with 'size' bytes.  return number of bytes actually read
    skip: #type (user: *void, n: s32) -> void #c_call; // skip the next 'n' bytes, or 'unget' the last -n bytes if negative
    eof:  #type (user: *void) -> s32 #c_call; // returns nonzero if we are at end of file/data
}

////////////////////////////////////
//
// 8-bits-per-channel interface
//
stbi_load_from_memory :: (buffer: *stbi_uc, len: s32, x: *s32, y: *s32, channels_in_file: *s32, desired_channels: s32) -> *stbi_uc #foreign stb_image;
stbi_load_from_callbacks :: (clbk: *stbi_io_callbacks, user: *void, x: *s32, y: *s32, channels_in_file: *s32, desired_channels: s32) -> *stbi_uc #foreign stb_image;

stbi_load :: (filename: *u8, x: *s32, y: *s32, channels_in_file: *s32, desired_channels: s32) -> *stbi_uc #foreign stb_image;
stbi_load_from_file :: (f: *FILE, x: *s32, y: *s32, channels_in_file: *s32, desired_channels: s32) -> *stbi_uc #foreign stb_image;

stbi_load_gif_from_memory :: (buffer: *stbi_uc, len: s32, delays: **s32, x: *s32, y: *s32, z: *s32, comp: *s32, req_comp: s32) -> *stbi_uc #foreign stb_image;

////////////////////////////////////
//
// 16-bits-per-channel interface
//
stbi_load_16_from_memory :: (buffer: *stbi_uc, len: s32, x: *s32, y: *s32, channels_in_file: *s32, desired_channels: s32) -> *stbi_us #foreign stb_image;
stbi_load_16_from_callbacks :: (clbk: *stbi_io_callbacks, user: *void, x: *s32, y: *s32, channels_in_file: *s32, desired_channels: s32) -> *stbi_us #foreign stb_image;

stbi_load_16 :: (filename: *u8, x: *s32, y: *s32, channels_in_file: *s32, desired_channels: s32) -> *stbi_us #foreign stb_image;
stbi_load_from_file_16 :: (f: *FILE, x: *s32, y: *s32, channels_in_file: *s32, desired_channels: s32) -> *stbi_us #foreign stb_image;

stbi_loadf_from_memory :: (buffer: *stbi_uc, len: s32, x: *s32, y: *s32, channels_in_file: *s32, desired_channels: s32) -> *float #foreign stb_image;
stbi_loadf_from_callbacks :: (clbk: *stbi_io_callbacks, user: *void, x: *s32, y: *s32, channels_in_file: *s32, desired_channels: s32) -> *float #foreign stb_image;

stbi_loadf :: (filename: *u8, x: *s32, y: *s32, channels_in_file: *s32, desired_channels: s32) -> *float #foreign stb_image;
stbi_loadf_from_file :: (f: *FILE, x: *s32, y: *s32, channels_in_file: *s32, desired_channels: s32) -> *float #foreign stb_image;

stbi_hdr_to_ldr_gamma :: (gamma: float) -> void #foreign stb_image;
stbi_hdr_to_ldr_scale :: (scale: float) -> void #foreign stb_image;

stbi_ldr_to_hdr_gamma :: (gamma: float) -> void #foreign stb_image;
stbi_ldr_to_hdr_scale :: (scale: float) -> void #foreign stb_image;

// stbi_is_hdr is always defined, but always returns false if STBI_NO_HDR
stbi_is_hdr_from_callbacks :: (clbk: *stbi_io_callbacks, user: *void) -> s32 #foreign stb_image;
stbi_is_hdr_from_memory :: (buffer: *stbi_uc, len: s32) -> s32 #foreign stb_image;

stbi_is_hdr :: (filename: *u8) -> s32 #foreign stb_image;
stbi_is_hdr_from_file :: (f: *FILE) -> s32 #foreign stb_image;

// get a VERY brief reason for failure
// on most compilers (and ALL modern mainstream compilers) this is threadsafe
stbi_failure_reason :: () -> *u8 #foreign stb_image;

// free the loaded image -- this is just free()
stbi_image_free :: (retval_from_stbi_load: *void) -> void #foreign stb_image;

// get image dimensions & components without fully decoding
stbi_info_from_memory :: (buffer: *stbi_uc, len: s32, x: *s32, y: *s32, comp: *s32) -> s32 #foreign stb_image;
stbi_info_from_callbacks :: (clbk: *stbi_io_callbacks, user: *void, x: *s32, y: *s32, comp: *s32) -> s32 #foreign stb_image;
stbi_is_16_bit_from_memory :: (buffer: *stbi_uc, len: s32) -> s32 #foreign stb_image;
stbi_is_16_bit_from_callbacks :: (clbk: *stbi_io_callbacks, user: *void) -> s32 #foreign stb_image;

stbi_info :: (filename: *u8, x: *s32, y: *s32, comp: *s32) -> s32 #foreign stb_image;
stbi_info_from_file :: (f: *FILE, x: *s32, y: *s32, comp: *s32) -> s32 #foreign stb_image;
stbi_is_16_bit :: (filename: *u8) -> s32 #foreign stb_image;
stbi_is_16_bit_from_file :: (f: *FILE) -> s32 #foreign stb_image;

// for image formats that explicitly notate that they have premultiplied alpha,
// we just return the colors as stored in the file. set this flag to force
// unpremultiplication. results are undefined if the unpremultiply overflow.
stbi_set_unpremultiply_on_load :: (flag_true_if_should_unpremultiply: s32) -> void #foreign stb_image;

// indicate whether we should process iphone images back to canonical format,
// or just pass them through "as-is"
stbi_convert_iphone_png_to_rgb :: (flag_true_if_should_convert: s32) -> void #foreign stb_image;

// flip the image vertically, so the first pixel in the output array is the bottom left
stbi_set_flip_vertically_on_load :: (flag_true_if_should_flip: s32) -> void #foreign stb_image;

// as above, but only applies to images loaded on the thread that calls the function
// this function is only available if your compiler supports thread-local variables;
// calling it will fail to link if your compiler doesn't
stbi_set_unpremultiply_on_load_thread :: (flag_true_if_should_unpremultiply: s32) -> void #foreign stb_image;
stbi_convert_iphone_png_to_rgb_thread :: (flag_true_if_should_convert: s32) -> void #foreign stb_image;
stbi_set_flip_vertically_on_load_thread :: (flag_true_if_should_flip: s32) -> void #foreign stb_image;

// ZLIB client - used by PNG, available for other purposes
stbi_zlib_decode_malloc_guesssize :: (buffer: *u8, len: s32, initial_size: s32, outlen: *s32) -> *u8 #foreign stb_image;
stbi_zlib_decode_malloc_guesssize_headerflag :: (buffer: *u8, len: s32, initial_size: s32, outlen: *s32, parse_header: s32) -> *u8 #foreign stb_image;
stbi_zlib_decode_malloc :: (buffer: *u8, len: s32, outlen: *s32) -> *u8 #foreign stb_image;
stbi_zlib_decode_buffer :: (obuffer: *u8, olen: s32, ibuffer: *u8, ilen: s32) -> s32 #foreign stb_image;

stbi_zlib_decode_noheader_malloc :: (buffer: *u8, len: s32, outlen: *s32) -> *u8 #foreign stb_image;
stbi_zlib_decode_noheader_buffer :: (obuffer: *u8, olen: s32, ibuffer: *u8, ilen: s32) -> s32 #foreign stb_image;

#scope_file

#import "Basic"; // For assert

stb_image :: #library "lib/stb_image";

#run {
    {
        info := type_info(stbi_io_callbacks);
        for info.members {
            if it.name == {
                case "read";
                    assert(it.offset_in_bytes == 0, "stbi_io_callbacks.read has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "stbi_io_callbacks.read has unexpected size % instead of 8", it.type.runtime_size);
                case "skip";
                    assert(it.offset_in_bytes == 8, "stbi_io_callbacks.skip has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "stbi_io_callbacks.skip has unexpected size % instead of 8", it.type.runtime_size);
                case "eof";
                    assert(it.offset_in_bytes == 16, "stbi_io_callbacks.eof has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "stbi_io_callbacks.eof has unexpected size % instead of 8", it.type.runtime_size);
            }
        }
        assert(size_of(stbi_io_callbacks) == 24, "stbi_io_callbacks has size % instead of 24", size_of(stbi_io_callbacks));
    }
}

