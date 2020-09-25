# Delete non-deforming bones
rig = bpy.context.object
bones = rig.data.bones.keys()
bones = [b.name for b in bpy.context.selected_pose_bones]

for ob in bpy.data.objects:
    if (ob.type != 'MESH') or (not ob.vertex_groups):
        continue
    for b in bones.copy():
        if b in ob.vertex_groups.keys():
            bones.remove(b)
    # if not bones:
    #     break
else:
    if "Keep bones, just select them":
        for b in rig.data.bones:
            b.select = b.name in bones
    else:
        bpy.ops.object.mode_set(mode='EDIT')
        for b in bones:
            rig.data.edit_bones.remove(rig.data.edit_bones[b])
        bpy.ops.object.mode_set(mode='OBJECT')
