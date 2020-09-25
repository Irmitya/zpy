# Change single armature constraints to parenting
import bpy
pars = dict()

bpy.ops.object.mode_set(mode='POSE')
for b in bpy.context.object.pose.bones:
    for c in b.constraints:
        if c.type == 'ARMATURE':
            if len(c.targets) == 1:
                pars[b.name] = c.targets[0].subtarget
                b.constraints.remove(c)

bpy.ops.object.mode_set(mode='EDIT')
for (nameb, namep) in pars.items():
    b = bpy.context.object.data.edit_bones[nameb]
    p = bpy.context.object.data.edit_bones[namep]
    b.parent = p

bpy.ops.object.mode_set(mode='POSE')
