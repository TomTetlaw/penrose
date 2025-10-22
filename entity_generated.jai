g_write_entity :: (base_entity: *Entity, file: *File)
{
  if base_entity.info.type ==
  {
    case Entity; entity := g_downcast(base_entity, Entity);
file_print_line(file, "entity.name: %", entity.name);
if entity.mesh file_print_line(file, "entity.mesh: %", entity.mesh.path);
file_print_v3(file, "entity.position", entity.position);
file_print_v3(file, "entity.rotation", entity.rotation);
file_print_v3(file, "entity.scale", entity.scale);
if entity.material file_print_line(file, "entity.material: %", entity.material.name);
file_print_v4(file, "entity.colour", entity.colour);
file_print_line(file, "entity.collision.type: %", entity.collision.type);
file_print_line(file, "entity.collision.enabled: %", entity.collision.enabled);
file_print_v3(file, "entity.collision.box_scale", entity.collision.box_scale);
file_print_line(file, "entity.collision.capsule_radius: %", entity.collision.capsule_radius);
file_print_line(file, "entity.collision.capsule_height: %", entity.collision.capsule_height);

case Prop; entity := g_downcast(base_entity, Prop);
file_print_line(file, "entity.base.name: %", entity.base.name);
if entity.base.mesh file_print_line(file, "entity.base.mesh: %", entity.base.mesh.path);
file_print_v3(file, "entity.base.position", entity.base.position);
file_print_v3(file, "entity.base.rotation", entity.base.rotation);
file_print_v3(file, "entity.base.scale", entity.base.scale);
if entity.base.material file_print_line(file, "entity.base.material: %", entity.base.material.name);
file_print_v4(file, "entity.base.colour", entity.base.colour);
file_print_line(file, "entity.base.collision.type: %", entity.base.collision.type);
file_print_line(file, "entity.base.collision.enabled: %", entity.base.collision.enabled);
file_print_v3(file, "entity.base.collision.box_scale", entity.base.collision.box_scale);
file_print_line(file, "entity.base.collision.capsule_radius: %", entity.base.collision.capsule_radius);
file_print_line(file, "entity.base.collision.capsule_height: %", entity.base.collision.capsule_height);

case PlayerStart; entity := g_downcast(base_entity, PlayerStart);
file_print_line(file, "entity.base.name: %", entity.base.name);
if entity.base.mesh file_print_line(file, "entity.base.mesh: %", entity.base.mesh.path);
file_print_v3(file, "entity.base.position", entity.base.position);
file_print_v3(file, "entity.base.rotation", entity.base.rotation);
file_print_v3(file, "entity.base.scale", entity.base.scale);
if entity.base.material file_print_line(file, "entity.base.material: %", entity.base.material.name);
file_print_v4(file, "entity.base.colour", entity.base.colour);
file_print_line(file, "entity.base.collision.type: %", entity.base.collision.type);
file_print_line(file, "entity.base.collision.enabled: %", entity.base.collision.enabled);
file_print_v3(file, "entity.base.collision.box_scale", entity.base.collision.box_scale);
file_print_line(file, "entity.base.collision.capsule_radius: %", entity.base.collision.capsule_radius);
file_print_line(file, "entity.base.collision.capsule_height: %", entity.base.collision.capsule_height);

case Player; entity := g_downcast(base_entity, Player);
file_print_line(file, "entity.base.name: %", entity.base.name);
if entity.base.mesh file_print_line(file, "entity.base.mesh: %", entity.base.mesh.path);
file_print_v3(file, "entity.base.position", entity.base.position);
file_print_v3(file, "entity.base.rotation", entity.base.rotation);
file_print_v3(file, "entity.base.scale", entity.base.scale);
if entity.base.material file_print_line(file, "entity.base.material: %", entity.base.material.name);
file_print_v4(file, "entity.base.colour", entity.base.colour);
file_print_line(file, "entity.base.collision.type: %", entity.base.collision.type);
file_print_line(file, "entity.base.collision.enabled: %", entity.base.collision.enabled);
file_print_v3(file, "entity.base.collision.box_scale", entity.base.collision.box_scale);
file_print_line(file, "entity.base.collision.capsule_radius: %", entity.base.collision.capsule_radius);
file_print_line(file, "entity.base.collision.capsule_height: %", entity.base.collision.capsule_height);
file_print_line(file, "entity.move_speed: %", entity.move_speed);

case UsableObject; entity := g_downcast(base_entity, UsableObject);
file_print_line(file, "entity.base.name: %", entity.base.name);
if entity.base.mesh file_print_line(file, "entity.base.mesh: %", entity.base.mesh.path);
file_print_v3(file, "entity.base.position", entity.base.position);
file_print_v3(file, "entity.base.rotation", entity.base.rotation);
file_print_v3(file, "entity.base.scale", entity.base.scale);
if entity.base.material file_print_line(file, "entity.base.material: %", entity.base.material.name);
file_print_v4(file, "entity.base.colour", entity.base.colour);
file_print_line(file, "entity.base.collision.type: %", entity.base.collision.type);
file_print_line(file, "entity.base.collision.enabled: %", entity.base.collision.enabled);
file_print_v3(file, "entity.base.collision.box_scale", entity.base.collision.box_scale);
file_print_line(file, "entity.base.collision.capsule_radius: %", entity.base.collision.capsule_radius);
file_print_line(file, "entity.base.collision.capsule_height: %", entity.base.collision.capsule_height);
file_print_v2(file, "entity.pixel_size", entity.pixel_size);
file_print_line(file, "entity.type: %", entity.type);
file_print_line(file, "entity.custom_id: %", entity.custom_id);
file_print_line(file, "entity.target_name: %", entity.target_name);

case DynamicLight; entity := g_downcast(base_entity, DynamicLight);
file_print_line(file, "entity.base.name: %", entity.base.name);
if entity.base.mesh file_print_line(file, "entity.base.mesh: %", entity.base.mesh.path);
file_print_v3(file, "entity.base.position", entity.base.position);
file_print_v3(file, "entity.base.rotation", entity.base.rotation);
file_print_v3(file, "entity.base.scale", entity.base.scale);
if entity.base.material file_print_line(file, "entity.base.material: %", entity.base.material.name);
file_print_v4(file, "entity.base.colour", entity.base.colour);
file_print_line(file, "entity.base.collision.type: %", entity.base.collision.type);
file_print_line(file, "entity.base.collision.enabled: %", entity.base.collision.enabled);
file_print_v3(file, "entity.base.collision.box_scale", entity.base.collision.box_scale);
file_print_line(file, "entity.base.collision.capsule_radius: %", entity.base.collision.capsule_radius);
file_print_line(file, "entity.base.collision.capsule_height: %", entity.base.collision.capsule_height);
file_print_v3(file, "entity.light_info.position", entity.light_info.position);
file_print_line(file, "entity.light_info.type: %", entity.light_info.type);
file_print_v3(file, "entity.light_info.colour", entity.light_info.colour);
file_print_line(file, "entity.light_info.brightness: %", entity.light_info.brightness);
file_print_v3(file, "entity.light_info.dir", entity.light_info.dir);
file_print_line(file, "entity.light_info.pad0: %", entity.light_info.pad0);
file_print_line(file, "entity.light_info.inner_radius: %", entity.light_info.inner_radius);
file_print_line(file, "entity.light_info.outer_radius: %", entity.light_info.outer_radius);
file_print_v2(file, "entity.light_info.pad1", entity.light_info.pad1);

case Sphere; entity := g_downcast(base_entity, Sphere);
file_print_line(file, "entity.base.name: %", entity.base.name);
if entity.base.mesh file_print_line(file, "entity.base.mesh: %", entity.base.mesh.path);
file_print_v3(file, "entity.base.position", entity.base.position);
file_print_v3(file, "entity.base.rotation", entity.base.rotation);
file_print_v3(file, "entity.base.scale", entity.base.scale);
if entity.base.material file_print_line(file, "entity.base.material: %", entity.base.material.name);
file_print_v4(file, "entity.base.colour", entity.base.colour);
file_print_line(file, "entity.base.collision.type: %", entity.base.collision.type);
file_print_line(file, "entity.base.collision.enabled: %", entity.base.collision.enabled);
file_print_v3(file, "entity.base.collision.box_scale", entity.base.collision.box_scale);
file_print_line(file, "entity.base.collision.capsule_radius: %", entity.base.collision.capsule_radius);
file_print_line(file, "entity.base.collision.capsule_height: %", entity.base.collision.capsule_height);

case Door; entity := g_downcast(base_entity, Door);
file_print_line(file, "entity.base.name: %", entity.base.name);
if entity.base.mesh file_print_line(file, "entity.base.mesh: %", entity.base.mesh.path);
file_print_v3(file, "entity.base.position", entity.base.position);
file_print_v3(file, "entity.base.rotation", entity.base.rotation);
file_print_v3(file, "entity.base.scale", entity.base.scale);
if entity.base.material file_print_line(file, "entity.base.material: %", entity.base.material.name);
file_print_v4(file, "entity.base.colour", entity.base.colour);
file_print_line(file, "entity.base.collision.type: %", entity.base.collision.type);
file_print_line(file, "entity.base.collision.enabled: %", entity.base.collision.enabled);
file_print_v3(file, "entity.base.collision.box_scale", entity.base.collision.box_scale);
file_print_line(file, "entity.base.collision.capsule_radius: %", entity.base.collision.capsule_radius);
file_print_line(file, "entity.base.collision.capsule_height: %", entity.base.collision.capsule_height);
file_print_line(file, "entity.open: %", entity.open);


  }
}
g_read_entity_member :: (base_entity: *Entity, key: string, value: string)
{
  if base_entity.info.type ==
  {
    case Entity; entity := g_downcast(base_entity, Entity);
if key == "entity.name" entity.name = value;
if key == "entity.mesh" entity.mesh = dr_get_mesh(value);
if key == "entity.position" entity.position = parse_v3(value);
if key == "entity.rotation" entity.rotation = parse_v3(value);
if key == "entity.scale" entity.scale = parse_v3(value);
if key == "entity.material" entity.material = dr_get_material(value);
if key == "entity.colour" entity.colour = parse_v4(value);
if key == "entity.collision.type" entity.collision.type = parse_enum(*value, type_of(entity.collision.type));
if key == "entity.collision.enabled" entity.collision.enabled = parse_bool(*value);
if key == "entity.collision.box_scale" entity.collision.box_scale = parse_v3(value);
if key == "entity.collision.capsule_radius" entity.collision.capsule_radius = parse_float(*value);
if key == "entity.collision.capsule_height" entity.collision.capsule_height = parse_float(*value);

case Prop; entity := g_downcast(base_entity, Prop);
if key == "entity.base.name" entity.base.name = value;
if key == "entity.base.mesh" entity.base.mesh = dr_get_mesh(value);
if key == "entity.base.position" entity.base.position = parse_v3(value);
if key == "entity.base.rotation" entity.base.rotation = parse_v3(value);
if key == "entity.base.scale" entity.base.scale = parse_v3(value);
if key == "entity.base.material" entity.base.material = dr_get_material(value);
if key == "entity.base.colour" entity.base.colour = parse_v4(value);
if key == "entity.base.collision.type" entity.base.collision.type = parse_enum(*value, type_of(entity.base.collision.type));
if key == "entity.base.collision.enabled" entity.base.collision.enabled = parse_bool(*value);
if key == "entity.base.collision.box_scale" entity.base.collision.box_scale = parse_v3(value);
if key == "entity.base.collision.capsule_radius" entity.base.collision.capsule_radius = parse_float(*value);
if key == "entity.base.collision.capsule_height" entity.base.collision.capsule_height = parse_float(*value);

case PlayerStart; entity := g_downcast(base_entity, PlayerStart);
if key == "entity.base.name" entity.base.name = value;
if key == "entity.base.mesh" entity.base.mesh = dr_get_mesh(value);
if key == "entity.base.position" entity.base.position = parse_v3(value);
if key == "entity.base.rotation" entity.base.rotation = parse_v3(value);
if key == "entity.base.scale" entity.base.scale = parse_v3(value);
if key == "entity.base.material" entity.base.material = dr_get_material(value);
if key == "entity.base.colour" entity.base.colour = parse_v4(value);
if key == "entity.base.collision.type" entity.base.collision.type = parse_enum(*value, type_of(entity.base.collision.type));
if key == "entity.base.collision.enabled" entity.base.collision.enabled = parse_bool(*value);
if key == "entity.base.collision.box_scale" entity.base.collision.box_scale = parse_v3(value);
if key == "entity.base.collision.capsule_radius" entity.base.collision.capsule_radius = parse_float(*value);
if key == "entity.base.collision.capsule_height" entity.base.collision.capsule_height = parse_float(*value);

case Player; entity := g_downcast(base_entity, Player);
if key == "entity.base.name" entity.base.name = value;
if key == "entity.base.mesh" entity.base.mesh = dr_get_mesh(value);
if key == "entity.base.position" entity.base.position = parse_v3(value);
if key == "entity.base.rotation" entity.base.rotation = parse_v3(value);
if key == "entity.base.scale" entity.base.scale = parse_v3(value);
if key == "entity.base.material" entity.base.material = dr_get_material(value);
if key == "entity.base.colour" entity.base.colour = parse_v4(value);
if key == "entity.base.collision.type" entity.base.collision.type = parse_enum(*value, type_of(entity.base.collision.type));
if key == "entity.base.collision.enabled" entity.base.collision.enabled = parse_bool(*value);
if key == "entity.base.collision.box_scale" entity.base.collision.box_scale = parse_v3(value);
if key == "entity.base.collision.capsule_radius" entity.base.collision.capsule_radius = parse_float(*value);
if key == "entity.base.collision.capsule_height" entity.base.collision.capsule_height = parse_float(*value);
if key == "entity.move_speed" entity.move_speed = parse_float(*value);

case UsableObject; entity := g_downcast(base_entity, UsableObject);
if key == "entity.base.name" entity.base.name = value;
if key == "entity.base.mesh" entity.base.mesh = dr_get_mesh(value);
if key == "entity.base.position" entity.base.position = parse_v3(value);
if key == "entity.base.rotation" entity.base.rotation = parse_v3(value);
if key == "entity.base.scale" entity.base.scale = parse_v3(value);
if key == "entity.base.material" entity.base.material = dr_get_material(value);
if key == "entity.base.colour" entity.base.colour = parse_v4(value);
if key == "entity.base.collision.type" entity.base.collision.type = parse_enum(*value, type_of(entity.base.collision.type));
if key == "entity.base.collision.enabled" entity.base.collision.enabled = parse_bool(*value);
if key == "entity.base.collision.box_scale" entity.base.collision.box_scale = parse_v3(value);
if key == "entity.base.collision.capsule_radius" entity.base.collision.capsule_radius = parse_float(*value);
if key == "entity.base.collision.capsule_height" entity.base.collision.capsule_height = parse_float(*value);
if key == "entity.pixel_size" entity.pixel_size = parse_v2(value);
if key == "entity.type" entity.type = parse_enum(*value, type_of(entity.type));
if key == "entity.custom_id" entity.custom_id = parse_enum(*value, type_of(entity.custom_id));
if key == "entity.target_name" entity.target_name = value;

case DynamicLight; entity := g_downcast(base_entity, DynamicLight);
if key == "entity.base.name" entity.base.name = value;
if key == "entity.base.mesh" entity.base.mesh = dr_get_mesh(value);
if key == "entity.base.position" entity.base.position = parse_v3(value);
if key == "entity.base.rotation" entity.base.rotation = parse_v3(value);
if key == "entity.base.scale" entity.base.scale = parse_v3(value);
if key == "entity.base.material" entity.base.material = dr_get_material(value);
if key == "entity.base.colour" entity.base.colour = parse_v4(value);
if key == "entity.base.collision.type" entity.base.collision.type = parse_enum(*value, type_of(entity.base.collision.type));
if key == "entity.base.collision.enabled" entity.base.collision.enabled = parse_bool(*value);
if key == "entity.base.collision.box_scale" entity.base.collision.box_scale = parse_v3(value);
if key == "entity.base.collision.capsule_radius" entity.base.collision.capsule_radius = parse_float(*value);
if key == "entity.base.collision.capsule_height" entity.base.collision.capsule_height = parse_float(*value);
if key == "entity.light_info.position" entity.light_info.position = parse_v3(value);
if key == "entity.light_info.type" entity.light_info.type = parse_enum(*value, type_of(entity.light_info.type));
if key == "entity.light_info.colour" entity.light_info.colour = parse_v3(value);
if key == "entity.light_info.brightness" entity.light_info.brightness = parse_float(*value);
if key == "entity.light_info.dir" entity.light_info.dir = parse_v3(value);
if key == "entity.light_info.pad0" entity.light_info.pad0 = parse_float(*value);
if key == "entity.light_info.inner_radius" entity.light_info.inner_radius = parse_float(*value);
if key == "entity.light_info.outer_radius" entity.light_info.outer_radius = parse_float(*value);
if key == "entity.light_info.pad1" entity.light_info.pad1 = parse_v2(value);

case Sphere; entity := g_downcast(base_entity, Sphere);
if key == "entity.base.name" entity.base.name = value;
if key == "entity.base.mesh" entity.base.mesh = dr_get_mesh(value);
if key == "entity.base.position" entity.base.position = parse_v3(value);
if key == "entity.base.rotation" entity.base.rotation = parse_v3(value);
if key == "entity.base.scale" entity.base.scale = parse_v3(value);
if key == "entity.base.material" entity.base.material = dr_get_material(value);
if key == "entity.base.colour" entity.base.colour = parse_v4(value);
if key == "entity.base.collision.type" entity.base.collision.type = parse_enum(*value, type_of(entity.base.collision.type));
if key == "entity.base.collision.enabled" entity.base.collision.enabled = parse_bool(*value);
if key == "entity.base.collision.box_scale" entity.base.collision.box_scale = parse_v3(value);
if key == "entity.base.collision.capsule_radius" entity.base.collision.capsule_radius = parse_float(*value);
if key == "entity.base.collision.capsule_height" entity.base.collision.capsule_height = parse_float(*value);

case Door; entity := g_downcast(base_entity, Door);
if key == "entity.base.name" entity.base.name = value;
if key == "entity.base.mesh" entity.base.mesh = dr_get_mesh(value);
if key == "entity.base.position" entity.base.position = parse_v3(value);
if key == "entity.base.rotation" entity.base.rotation = parse_v3(value);
if key == "entity.base.scale" entity.base.scale = parse_v3(value);
if key == "entity.base.material" entity.base.material = dr_get_material(value);
if key == "entity.base.colour" entity.base.colour = parse_v4(value);
if key == "entity.base.collision.type" entity.base.collision.type = parse_enum(*value, type_of(entity.base.collision.type));
if key == "entity.base.collision.enabled" entity.base.collision.enabled = parse_bool(*value);
if key == "entity.base.collision.box_scale" entity.base.collision.box_scale = parse_v3(value);
if key == "entity.base.collision.capsule_radius" entity.base.collision.capsule_radius = parse_float(*value);
if key == "entity.base.collision.capsule_height" entity.base.collision.capsule_height = parse_float(*value);
if key == "entity.open" entity.open = parse_bool(*value);


  }
}
g_entity_imgui :: (base_entity: *Entity)
{
  if base_entity.info.type ==
  {
    case Entity; entity := g_downcast(base_entity, Entity);
string_property("entity.name", *entity.name);
vec3_property("entity.position", *entity.position);
vec3_property("entity.rotation", *entity.rotation);
vec3_property("entity.scale", *entity.scale);
material_property("entity.material", *entity.material);
vec4_property("entity.colour", *entity.colour);
enum_property("entity.collision.type", *entity.collision.type);
ImGui.Checkbox("entity.collision.enabled", *entity.collision.enabled);
vec3_property("entity.collision.box_scale", *entity.collision.box_scale);
ImGui.DragFloat("entity.collision.capsule_radius", *entity.collision.capsule_radius);
ImGui.DragFloat("entity.collision.capsule_height", *entity.collision.capsule_height);

case Prop; entity := g_downcast(base_entity, Prop);
string_property("entity.base.name", *entity.base.name);
vec3_property("entity.base.position", *entity.base.position);
vec3_property("entity.base.rotation", *entity.base.rotation);
vec3_property("entity.base.scale", *entity.base.scale);
material_property("entity.base.material", *entity.base.material);
vec4_property("entity.base.colour", *entity.base.colour);
enum_property("entity.base.collision.type", *entity.base.collision.type);
ImGui.Checkbox("entity.base.collision.enabled", *entity.base.collision.enabled);
vec3_property("entity.base.collision.box_scale", *entity.base.collision.box_scale);
ImGui.DragFloat("entity.base.collision.capsule_radius", *entity.base.collision.capsule_radius);
ImGui.DragFloat("entity.base.collision.capsule_height", *entity.base.collision.capsule_height);

case PlayerStart; entity := g_downcast(base_entity, PlayerStart);
string_property("entity.base.name", *entity.base.name);
vec3_property("entity.base.position", *entity.base.position);
vec3_property("entity.base.rotation", *entity.base.rotation);
vec3_property("entity.base.scale", *entity.base.scale);
material_property("entity.base.material", *entity.base.material);
vec4_property("entity.base.colour", *entity.base.colour);
enum_property("entity.base.collision.type", *entity.base.collision.type);
ImGui.Checkbox("entity.base.collision.enabled", *entity.base.collision.enabled);
vec3_property("entity.base.collision.box_scale", *entity.base.collision.box_scale);
ImGui.DragFloat("entity.base.collision.capsule_radius", *entity.base.collision.capsule_radius);
ImGui.DragFloat("entity.base.collision.capsule_height", *entity.base.collision.capsule_height);

case Player; entity := g_downcast(base_entity, Player);
string_property("entity.base.name", *entity.base.name);
vec3_property("entity.base.position", *entity.base.position);
vec3_property("entity.base.rotation", *entity.base.rotation);
vec3_property("entity.base.scale", *entity.base.scale);
material_property("entity.base.material", *entity.base.material);
vec4_property("entity.base.colour", *entity.base.colour);
enum_property("entity.base.collision.type", *entity.base.collision.type);
ImGui.Checkbox("entity.base.collision.enabled", *entity.base.collision.enabled);
vec3_property("entity.base.collision.box_scale", *entity.base.collision.box_scale);
ImGui.DragFloat("entity.base.collision.capsule_radius", *entity.base.collision.capsule_radius);
ImGui.DragFloat("entity.base.collision.capsule_height", *entity.base.collision.capsule_height);
ImGui.DragFloat("entity.move_speed", *entity.move_speed);

case UsableObject; entity := g_downcast(base_entity, UsableObject);
string_property("entity.base.name", *entity.base.name);
vec3_property("entity.base.position", *entity.base.position);
vec3_property("entity.base.rotation", *entity.base.rotation);
vec3_property("entity.base.scale", *entity.base.scale);
material_property("entity.base.material", *entity.base.material);
vec4_property("entity.base.colour", *entity.base.colour);
enum_property("entity.base.collision.type", *entity.base.collision.type);
ImGui.Checkbox("entity.base.collision.enabled", *entity.base.collision.enabled);
vec3_property("entity.base.collision.box_scale", *entity.base.collision.box_scale);
ImGui.DragFloat("entity.base.collision.capsule_radius", *entity.base.collision.capsule_radius);
ImGui.DragFloat("entity.base.collision.capsule_height", *entity.base.collision.capsule_height);
vec2_property("entity.pixel_size", *entity.pixel_size);
enum_property("entity.type", *entity.type);
enum_property("entity.custom_id", *entity.custom_id);
string_property("entity.target_name", *entity.target_name);

case DynamicLight; entity := g_downcast(base_entity, DynamicLight);
string_property("entity.base.name", *entity.base.name);
vec3_property("entity.base.position", *entity.base.position);
vec3_property("entity.base.rotation", *entity.base.rotation);
vec3_property("entity.base.scale", *entity.base.scale);
material_property("entity.base.material", *entity.base.material);
vec4_property("entity.base.colour", *entity.base.colour);
enum_property("entity.base.collision.type", *entity.base.collision.type);
ImGui.Checkbox("entity.base.collision.enabled", *entity.base.collision.enabled);
vec3_property("entity.base.collision.box_scale", *entity.base.collision.box_scale);
ImGui.DragFloat("entity.base.collision.capsule_radius", *entity.base.collision.capsule_radius);
ImGui.DragFloat("entity.base.collision.capsule_height", *entity.base.collision.capsule_height);
vec3_property("entity.light_info.position", *entity.light_info.position);
enum_property("entity.light_info.type", *entity.light_info.type);
vec3_property("entity.light_info.colour", *entity.light_info.colour);
ImGui.DragFloat("entity.light_info.brightness", *entity.light_info.brightness);
vec3_property("entity.light_info.dir", *entity.light_info.dir);
ImGui.DragFloat("entity.light_info.pad0", *entity.light_info.pad0);
ImGui.DragFloat("entity.light_info.inner_radius", *entity.light_info.inner_radius);
ImGui.DragFloat("entity.light_info.outer_radius", *entity.light_info.outer_radius);
vec2_property("entity.light_info.pad1", *entity.light_info.pad1);

case Sphere; entity := g_downcast(base_entity, Sphere);
string_property("entity.base.name", *entity.base.name);
vec3_property("entity.base.position", *entity.base.position);
vec3_property("entity.base.rotation", *entity.base.rotation);
vec3_property("entity.base.scale", *entity.base.scale);
material_property("entity.base.material", *entity.base.material);
vec4_property("entity.base.colour", *entity.base.colour);
enum_property("entity.base.collision.type", *entity.base.collision.type);
ImGui.Checkbox("entity.base.collision.enabled", *entity.base.collision.enabled);
vec3_property("entity.base.collision.box_scale", *entity.base.collision.box_scale);
ImGui.DragFloat("entity.base.collision.capsule_radius", *entity.base.collision.capsule_radius);
ImGui.DragFloat("entity.base.collision.capsule_height", *entity.base.collision.capsule_height);

case Door; entity := g_downcast(base_entity, Door);
string_property("entity.base.name", *entity.base.name);
vec3_property("entity.base.position", *entity.base.position);
vec3_property("entity.base.rotation", *entity.base.rotation);
vec3_property("entity.base.scale", *entity.base.scale);
material_property("entity.base.material", *entity.base.material);
vec4_property("entity.base.colour", *entity.base.colour);
enum_property("entity.base.collision.type", *entity.base.collision.type);
ImGui.Checkbox("entity.base.collision.enabled", *entity.base.collision.enabled);
vec3_property("entity.base.collision.box_scale", *entity.base.collision.box_scale);
ImGui.DragFloat("entity.base.collision.capsule_radius", *entity.base.collision.capsule_radius);
ImGui.DragFloat("entity.base.collision.capsule_height", *entity.base.collision.capsule_height);
ImGui.Checkbox("entity.open", *entity.open);


  }
}
g_create_entity_from_type_name :: (str: string) -> *Entity #expand
{
  if str ==
  {
    case "Entity"; return g_create_entity(Entity);
    case "Prop"; return g_create_entity(Prop);
    case "PlayerStart"; return g_create_entity(PlayerStart);
    case "Player"; return g_create_entity(Player);
    case "UsableObject"; return g_create_entity(UsableObject);
    case "DynamicLight"; return g_create_entity(DynamicLight);
    case "Sphere"; return g_create_entity(Sphere);
    case "Door"; return g_create_entity(Door);

case; return g_create_entity(Entity);
  }
}
g_entity_type_info := EntityTypeInfo.[
  .{"Entity", Entity, size_of(Entity)},
  .{"Prop", Prop, size_of(Prop)},
  .{"PlayerStart", PlayerStart, size_of(PlayerStart)},
  .{"Player", Player, size_of(Player)},
  .{"UsableObject", UsableObject, size_of(UsableObject)},
  .{"DynamicLight", DynamicLight, size_of(DynamicLight)},
  .{"Sphere", Sphere, size_of(Sphere)},
  .{"Door", Door, size_of(Door)},

];
