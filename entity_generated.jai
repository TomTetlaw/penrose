g_serialize_entity :: (base_entity: *Entity, file: *File)
{
  if base_entity.info.type ==
  {
    case Entity; entity := g_downcast(base_entity, Entity);
file_print_line(file, "entity.name: %", entity.name);
file_print_line(file, "entity.mesh", entity.mesh.path);
file_print_v3(file, "entity.position", entity.position);
file_print_v3(file, "entity.rotation", entity.rotation);
file_print_v3(file, "entity.scale", entity.scale);
entity.materialfile_print_v4(file, "entity.colour", entity.colour);
entity.collision
case UsableObject; entity := g_downcast(base_entity, UsableObject);
entity.basefile_print_v2(file, "entity.pixel_size", entity.pixel_size);

case Prop; entity := g_downcast(base_entity, Prop);
entity.base
case PlayerStart; entity := g_downcast(base_entity, PlayerStart);
entity.base
case Player; entity := g_downcast(base_entity, Player);
entity.basefile_print_line(file, "entity.move_speed: %", entity.move_speed);


  }
}
g_create_entity_from_type_name :: (str: string) -> *Entity #expand
{
  if str ==
  {
    case "Entity"; return g_create_entity(Entity);
    case "UsableObject"; return g_create_entity(UsableObject);
    case "Prop"; return g_create_entity(Prop);
    case "PlayerStart"; return g_create_entity(PlayerStart);
    case "Player"; return g_create_entity(Player);

case; return g_create_entity(Entity);
  }
}
g_entity_type_info := EntityTypeInfo.[
  .{"Entity", Entity, size_of(Entity)},
  .{"UsableObject", UsableObject, size_of(UsableObject)},
  .{"Prop", Prop, size_of(Prop)},
  .{"PlayerStart", PlayerStart, size_of(PlayerStart)},
  .{"Player", Player, size_of(Player)},

];
