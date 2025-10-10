g_create_entity_from_type_name :: (str: string) -> *Entity #expand
{
  if str ==
  {
    case "Entity"; #through;
    case; return g_create_entity(Entity);
    case "Player"; return g_create_entity(Player);
    case "Prop"; return g_create_entity(Prop);
    case "PlayerStart"; return g_create_entity(PlayerStart);

  }
}
g_entity_type_info := EntityTypeInfo.[
  .{"Entity", Entity},
  .{"Player", Player},
  .{"Prop", Prop},
  .{"PlayerStart", PlayerStart},

];
