bpy.ops.object.select_all(action='DESELECT')
for o in D.collections['Widgets'].all_objects:
    m = o.data
    if o.users < 2 and not o.children:
        o.select_set(True)
        C.view_layer.objects.active = o
        # print(o.name)
        # D.objects.remove(o)
        # print('\t', m.users, m.name)
        # if not m.users:
        #     D.meshes.remove(m)
