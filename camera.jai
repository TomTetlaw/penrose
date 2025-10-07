
CAM_Base :: struct
{
  position: Vec3F32;
  forward: Vec3F32;
  view: Mat4x4F32;
  projection: Mat4x4F32;
}

CAM_Player :: struct
{
  using #as base: CAM_Base;
  rotation_input: Vec2F32;
  yaw, pitch: F32;
  move_speed :: 25.0;
}

CAM_Editor :: struct
{
  using #as base: CAM_Base;
  rotation_input: Vec2F32;
  yaw, pitch: F32;
  move_speed :: 25.0;
}

cam_editor_update :: (camera: *CAM_Editor)
{
  if in_mouse_state(.Right)
  {
    RotationSmoothing :: 0.23;
    camera.rotation_input = lerp_v2(camera.rotation_input, in_input_state.mouse_delta, RotationSmoothing);
    camera.yaw += camera.rotation_input.x * TurnsFromDegrees * RotationSmoothing;
    camera.pitch += camera.rotation_input.y * TurnsFromDegrees * RotationSmoothing;
    camera.pitch = clamp(camera.pitch, -0.25, 0.25);
  }
  yaw := mat4x4f32_rotate(v3f32(0, 0, 1), camera.yaw);
  pitch := mat4x4f32_rotate(v3f32(0, 1, 0), camera.pitch);
  rotation := yaw * pitch;
  move_speed := camera.move_speed;
  if in_mouse_state(.Right)
  {
    movement: Vec3F32;
    movement.x = in_key_state(.W) - in_key_state(.S);
    movement.y = in_key_state(.A) - in_key_state(.D);
    movement.z = in_key_state(.E) - in_key_state(.Q);
    movement = rotate_v3(rotation, movement);
    camera.position += movement * move_speed * game_delta;
  }
  transform := mat4x4f32_translate(camera.position) * rotation;
  camera.view = mat4x4f32_inverse(transform);
  camera.forward = v3f32(camera.view.m[0][0], camera.view.m[0][1], camera.view.m[0][2]);
  aspect := dr_renderer.resolution.x / dr_renderer.resolution.y;
  camera.projection = mat4x4f32_perspective(0.125, aspect, 0.1, 2000.0);
}

cam_player_update :: (camera: *CAM_Player)
{
  RotationSmoothing :: 0.23;
  camera.rotation_input = lerp_v2(camera.rotation_input, in_input_state.mouse_delta, RotationSmoothing);
  camera.yaw += camera.rotation_input.x * TurnsFromDegrees * RotationSmoothing;
  camera.pitch += camera.rotation_input.y * TurnsFromDegrees * RotationSmoothing;
  camera.pitch = clamp(camera.pitch, -0.25, 0.25);
  yaw := mat4x4f32_rotate(v3f32(0, 0, 1), camera.yaw);
  pitch := mat4x4f32_rotate(v3f32(0, 1, 0), camera.pitch);
  rotation := yaw * pitch;
  move_speed := camera.move_speed;
  movement: Vec2F32;
  movement.x = in_key_state(.W) - in_key_state(.S);
  movement.y = in_key_state(.A) - in_key_state(.D);
  movement = rotate_v2(rotation, movement);
  camera.position += v3f32(movement, 0.0) * move_speed * game_delta;
  transform := mat4x4f32_translate(camera.position) * rotation;
  camera.view = mat4x4f32_inverse(transform);
  camera.forward = v3f32(camera.view.m[0][0], camera.view.m[0][1], camera.view.m[0][2]);
  aspect := dr_renderer.resolution.x / dr_renderer.resolution.y;
  camera.projection = mat4x4f32_perspective(0.125, aspect, 0.1, 2000.0);
}