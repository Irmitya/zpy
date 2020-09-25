for b in bpy.context.object.pose.bones:
    for index, value in enumerate(b.rigify_parameters.tweak_layers):
        if index > 4:
            b.rigify_parameters.tweak_layers[index - 1] = value
    for index, value in enumerate(b.rigify_parameters.fk_layers):
        if index > 4:
            b.rigify_parameters.fk_layers[index - 1] = value
