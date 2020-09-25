import bpy
ob = bpy.context.object

abc = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
prefix = 'pJCM'
keys = ob.data.shape_keys.key_blocks

shapes = sorted(k for k in keys.keys() if k.startswith(prefix))
while shapes:
    current_shape = shapes.pop()
    index = keys.find(current_shape)
    ind = ob.active_shape_key_index
    ob.active_shape_key_index = index
    bpy.ops.object.shape_key_move(type='TOP')
    ob.active_shape_key_index = ind
