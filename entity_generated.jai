g_create_entity_from_type_name :: (str: string) -> *Entity #expand
{
  if str ==
  {
    case "Entity"; #through;
    case; return g_create_entity(Entity);
    case "Prop"; return g_create_entity(Prop);
    case "PlayerStart"; return g_create_entity(PlayerStart);
    case "Player"; return g_create_entity(Player);

  }
}
g_entity_type_info := EntityTypeInfo.[
  .{"Entity", Entity, size_of(Entity)},
  .{"Prop", Prop, size_of(Prop)},
  .{"PlayerStart", PlayerStart, size_of(PlayerStart)},
  .{"Player", Player, size_of(Player)},

];
