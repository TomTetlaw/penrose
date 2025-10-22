
CAM_Base :: struct
{
  position: V3;
  forward: V3;
  view: Mat4;
  projection: Mat4;
}

CAM_Player :: struct
{
  using #as base: CAM_Base;
  rotation_input: V2;
  yaw, pitch: float;
}

CAM_Editor :: struct
{
  using #as base: CAM_Base;
  rotation_input: V2;
  yaw, pitch: float;
  move_speed :: 25.0;
}

cam_editor_update :: (camera: *CAM_Editor)
{
  if in_mouse_state(.Right)
  {
    RotationSmoothing :: 0.23;
    camera.rotation_input = lerp(camera.rotation_input, in_input_state.mouse_delta, RotationSmoothing);
    camera.yaw += camera.rotation_input.x * TurnsFromDegrees * RotationSmoothing;
    camera.pitch += camera.rotation_input.y * TurnsFromDegrees * RotationSmoothing;
    camera.pitch = clamp(camera.pitch, -0.25, 0.25);
  }
  yaw := mat_rotate(v3(0, 0, 1), camera.yaw);
  pitch := mat_rotate(v3(0, 1, 0), camera.pitch);
  rotation := yaw * pitch;
  move_speed := camera.move_speed;
  if in_mouse_state(.Right)
  {
    movement: V3;
    movement.x = in_key_state(.W) - in_key_state(.S);
    movement.y = in_key_state(.A) - in_key_state(.D);
    movement.z = in_key_state(.E) - in_key_state(.Q);
    movement = rotate(rotation, movement);
    camera.position += movement * move_speed * real_delta;
  }
  transform := mat_translate(camera.position) * rotation;
  camera.view = mat_inverse(transform);
  camera.forward = v3(camera.view.m[0][0], camera.view.m[0][1], camera.view.m[0][2]);
  aspect := dr_renderer.resolution.x / dr_renderer.resolution.y;
  camera.projection = mat_perspective(0.125, aspect, 0.1, 2000.0);
}

cam_player_update :: (camera: *CAM_Player)
{
  RotationSmoothing :: 0.23;
  camera.rotation_input = lerp(camera.rotation_input, in_input_state.mouse_delta, RotationSmoothing);
  camera.yaw += camera.rotation_input.x * TurnsFromDegrees * RotationSmoothing;
  camera.pitch += camera.rotation_input.y * TurnsFromDegrees * RotationSmoothing;
  camera.pitch = clamp(camera.pitch, -0.25, 0.25);
  yaw := mat_rotate(v3(0, 0, 1), camera.yaw);
  pitch := mat_rotate(v3(0, 1, 0), camera.pitch);
  rotation := yaw * pitch;
  transform := mat_translate(camera.position) * rotation;
  camera.view = mat_inverse(transform);
  camera.forward = v3(camera.view.m[0][0], camera.view.m[0][1], camera.view.m[0][2]);
  aspect := dr_renderer.resolution.x / dr_renderer.resolution.y;
  camera.projection = mat_perspective(0.125, aspect, 0.1, 2000.0);
}