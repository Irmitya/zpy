import bpy
from bpy.props import (BoolProperty, FloatProperty, FloatVectorProperty, EnumProperty, StringProperty)
from mathutils import Matrix, Vector
from math import radians
from zpy import Get, Is, New, Set, utils

# Version check
is27 = bpy.app.version < (2, 80, 0)
is28 = not is27


class Constraint:
    """Global(constraint) properties that are called from the real Constraint"""

    no_target = (
        # Constraint types without a target,
        # so never generate objects for them
        'CAMERA_SOLVER',
        'FOLLOW_TRACK', 'OBJECT_SOLVER',  # 'target' is 'camera' object
        'LIMIT_LOCATION', 'LIMIT_ROTATION', 'LIMIT_SCALE',
        'MAINTAIN_VOLUME', 'TRANSFORM_CACHE',
    )

    curves = ('CLAMP_TO', 'SPLINE_IK', 'FOLLOW_PATH')

    class types:
        "Separate constraint types by category"

        motion_tracking = [
            ('CAMERA_SOLVER', "Camera Solver", "", 'CON_CAMERASOLVER', 1),
            ('FOLLOW_TRACK', "Follow Track", "", 'CON_FOLLOWTRACK', 2),
            ('OBJECT_SOLVER', "Object Solver", "", 'CON_OBJECTSOLVER', 3),
        ]
        transform = [
            ('COPY_LOCATION', "Copy Location",
                "Copy the location of a target (with an optional offset), "
                "so that they move together",
                'CON_LOCLIKE', 4),
            ('COPY_ROTATION', "Copy Rotation",
                "Copy the rotation of a target (with an optional offset), "
                "so that they rotate together",
                'CON_ROTLIKE', 5),
            ('COPY_SCALE', "Copy Scale",
                "Copy the scale factors of a target (with an optional offset), "
                "so that they are scaled by the same amount",
                'CON_SIZELIKE', 6),
            ('COPY_TRANSFORMS', "Copy Transforms",
                "Copy all the transformations of a target, "
                "so that they move together",
                'CON_TRANSLIKE', 7),
            ('LIMIT_DISTANCE', "Limit Distance",
                "Restrict movements to within a certain distance of a target "
                "(at the time of constraint evaluation only)",
                'CON_DISTLIMIT', 8),
            ('LIMIT_LOCATION', "Limit Location",
                "Restrict movement along each axis within given ranges",
                'CON_LOCLIMIT', 9),
            ('LIMIT_ROTATION', "Limit Rotation",
                "Restrict rotation along each axis within given ranges",
                'CON_ROTLIMIT', 10),
            ('LIMIT_SCALE', "Limit Scale",
                "Restrict scaling along each axis with given ranges",
                'CON_SIZELIMIT', 11),
            ('MAINTAIN_VOLUME', "Maintain Volume",
                "Compensate for scaling one axis by applying suitable "
                "scaling to the other two axes",
                'CON_SAMEVOL', 12),
            ('TRANSFORM', "Transformation",
                "Use one transform property from target to control "
                "another (or same) property on owner",
                'CON_TRANSFORM', 13),
            ('TRANSFORM_CACHE', "Transform Cache",
                "Look up the transformation matrix from an external file",
                'CON_TRANSFORM_CACHE', 14),
        ]
        tracking = [
            ('CLAMP_TO', "Clamp To",
                "Restrict movements to lie along a curve "
                "by remapping location along curve's longest axis",
                'CON_CLAMPTO', 15),
            ('DAMPED_TRACK', "Damped Track",
                "Point towards a target by performing the smallest rotation necessary",
                'CON_TRACKTO', 16),
            ('IK', "Inverse Kinematics",
                "Control a chain of bones by specifying the endpoint target (Bones only",
                'CON_KINEMATIC', 17),
            ('LOCKED_TRACK', "Locked Track",
                "Rotate around the specified ('locked') axis to point towards a target",
                'CON_LOCKTRACK', 18),
            ('SPLINE_IK', "Spline IK",
                "Align chain of bones along a curve (Bones only)",
                'CON_SPLINEIK', 19),
            ('STRETCH_TO', "Stretch To",
                "Stretch along Y-Axis to point towards a target",
                'CON_STRETCHTO', 20),
            ('TRACK_TO', "Track To",
                "Legacy tracking constraint prone to twisting artifacts",
                'CON_TRACKTO', 21),
        ]
        relationship = [
            ('ACTION', "Action",
                "Use transform property of target to "
                "look up pose for owner from an Action",
                'CON_ACTION', 22),
            *[('ARMATURE', "Armature",
                "Apply weight-blended transformation from "
                "multiple bones like the Armature modifier",
                'CON_ARMATURE', 23) for a in [is28] if a],  # Added in 2.80
            ('CHILD_OF', "Child Of",
                "Make target the 'detachable' parent of owner",
                'CON_CHILDOF', 24),
            ('FLOOR', "Floor",
                "Use position (and optionally rotation) of target to define a 'wall'"
                " or 'floor' that the owner can not cross",
                'CON_FLOOR', 25),
            ('FOLLOW_PATH', "Follow Path",
                "Use to animate an object/bone following a path",
                'CON_FOLLOWPATH', 26),
            ('PIVOT', "Pivot",
                "Change pivot point for transforms (buggy)",
                'CON_PIVOT', 27),
            *[('RIGID_BODY_JOINT', "Rigid Body Joint",
                "Use to define a Rigid Body Constraint (for Game Engine use only)",
                'CONSTRAINT_DATA', 28) for a in [is27] if a],  # removed in 2.80
            # ('SCRIPT', "Script",
            #  "Custom constraint(s) written in Python (Not yet implemented)",
            #  'CONSTRAINT_DATA', 29),  # removed around 2.6
            ('SHRINKWRAP', "Shrinkwrap", "Restrict movements to surface of target mesh",
                'CON_SHRINKWRAP', 30),
        ]

        # Custom
        child_of = [
            ('child_of_locrot', "Child of Loc + Rot",
                "Map to Location + Rotation separately", 'CON_TRANSLIKE', 52),
            ('child_of_location', "    Location",
                "Map only location", 'CON_LOCLIKE', 53),
            ('child_of_rotation', "    Rotation",
                "Map only rotation", 'CON_ROTLIKE', 54),
            ('child_of_scale', "    Scale", "Map only Scale", 'CON_SIZELIKE', 55),
        ]
        other = [
            ('ik_transforms', "IK Transforms",
                "IK with Lock Transforms", 'CON_KINEMATIC', 56),
            ('pivot', "Pivot",
                "Pivot around target point, like 3D Cursor (from rigify)", 'PIVOT_CURSOR', 57),
            ('track_rotation', "Damped Rotation",
                "Damped Track with Lock Rotation", 'CON_LOCKTRACK', 58),
            ('stretch_rotation', "Stretched Rotation",
                "Stretch To with Lock Rotation", 'CON_STRETCHTO', 59),
            # ('offset_transforms', "Offset Transforms",
            # "Transform constraint to \"add\" to the current transforms",
            # 'MOD_PARTICLE_INSTANCE', 59),
            # ('', "", "Other constraints"),
            # ('look_at', "(N/A) Look At",
                # "A series of offsetters and a Damped Track, "
                    # "to a single target to focus on"
                    # ".\n""Inspired by Meklab's SFM example / tutorial"
                    # ".\n""https://keythedrawist.tumblr.com/post/180848042768/some-constraint-tutorials",
                # 'CON_TRACKTO', 60),
        ]

    constraint_type_items = [
        ('', "Motion Tracking", ""), *types.motion_tracking,
        ('', "Transform", ""), *types.transform,
        ('', "Tracking", ""), *types.tracking,
        ('', "Relationship", ""), *types.relationship,
        # ('', "None", "Don't add a constraint, just create copy"),
        # Custom constraints
        ('', "Custom", ""),
        *types.child_of,
        *types.other,
    ]
    if is27: constraint_type_items = [
        (a, b, c[0]) for a, b, *c in constraint_type_items]

    # Store the ID of constraint types and custom types
    constraint_types = {x[0]: (x[1], x[2]) for x in (
        *types.motion_tracking,
        *types.transform,
        *types.tracking,
        *types.relationship,
    )}
    custom_types = {x[0]: (x[1], x[2]) for x in (
        *types.child_of,
        *types.other,
    )}


    ###############################################################################
    # region: Functions

    def get_track_axis(owner, axis):
        """
        Re-orient axis for objects\\
        Bones use the axis as-is but objects have a different Y-Z axis
        """

        if Is.object(owner) and axis.endswith('Y'):
            axis = axis[:-1] + 'Z'

        return axis

    def get_dict(src, prop):
        """
        Check if a custom prop is in src.\\
        If not, create it as a dict() and return it
        """

        if src.get(prop) is None:
            src[prop] = dict()

        return src[prop]

    def add_original_relation(context, src, original, use_local_transforms=True):
        """
        Remember the initial pose for new bones/objects\\
        Also remember what the original bone/object was\\
        Note: Only stored in the duplicate, nothing added to the original
        """
        if not hasattr(src, 'base_src'):
            return

        if src.rotation_mode in ('QUATERNION', 'AXIS_ANGLE'):
            euler = 'XYZ'
        else:
            euler = src.rotation_mode

        base = src.base_src
        base.is_duplicate = True
        if use_local_transforms:
            # Bone's default pose is the same as orignial's
            src.base_transforms.matrix_set(Get.matrix(src, basis=True), euler)
        base.target = original.id_data.name
        if Is.posebone(original):
            base.subtarget = original.name
        if original.id_data == src.id_data:
            base.target_self = True

        original.base_src.has_duplicates = True
        copies = original.base_src.copies
        src_hash = str(hash(src))
        if src_hash not in copies:
            copy = copies.add()
            copy.name = src_hash
            copy.label = src.id_data.name.split('||', 1)[0]
        else:
            copy = copies[src_hash]

        copy.target = src.id_data.name
        if Is.posebone(src):
            copy.subtarget = src.name

        return

        # if hasattr(src.id_data, 'constraints_extra'):
            # con_list = src.id_data.constraints_extra
            # if src.name in con_list:
                # con_entry = con_list[src.name]
            # else:
                # con_entry = con_list.add()
                # con_entry.name = src.name

            # con_self = con_entry.self
            # # con_self.bone = Is.posebone(src)
            # con_self['base_matrix'] = Get.matrix(src)

            # con_original = con_entry.original
            # con_original.target = original.id_data.name
            # if Is.posebone(original):
                # con_original.subtarget = original.name

            # # Add copy to the original
            # con_list = original.id_data.constraints_extra
            # if original.name in con_list:
                # con_entry = con_list[original.name]
            # else:
                # con_entry = con_list.add()
                # con_entry.name = original.name

            # con_copies = con_entry.copies
            # cc_entry = con_copies.add()
            # cc_entry.name = utils.string(
            # src.id_data.name, '-', src.name if Is.posebone(src) else '')

            # cc_entry.target = src.id_data.name
            # if Is.posebone(src):
                # cc_entry.subtarget = src.name
        # else:
            # con_list = Constraint.get_dict(src.id_data, 'constraints_extra')
            # con_entry = Constraint.get_dict(con_list, src.name)

            # con_entry['self'] = dict(
                # # bone=Is.posebone(src),
                # base_matrix=Get.matrix(src),
            # )
            # con_entry['original'] = dict(
                # target=original.id_data.name,
                # subtarget=original.name
                # if Is.posebone(original) else '',
            # )

            # # Add copy to the original
            # con_list = Constraint.get_dict(original.id_data, 'constraints_extra')
            # con_entry = Constraint.get_dict(con_list, original.name)

            # con_copies = Constraint.get_dict(con_entry, 'copies')
            # con_copies[utils.string(
            # src.id_data.name, '-', src.name if Is.posebone(src) else ''
            # )] = dict(
                # target=src.id_data.name,
                # subtarget=src.name
                # if Is.posebone(src) else '',
            # )

    # def add_constraint_relation(context, constraint, owner, target):
        # """
        # Add custom property to owner's object, to mark constraint information
        # """

        # # for src in (owner, target):
            # # if not src:
            #     # continue

            # # if not hasattr(src, 'base_src'):
            #     # continue

            # # continue

            # # owner_id = owner.id_data.name
            # # owner_bone = owner.name if Is.posebone(owner) else ''
            # # target_id = target.id_data.name if target else ''
            # # target_bone = target.name if Is.posebone(target) else ''

            # # if hasattr(src.id_data, 'constraints_extra'):
            #     # con_list = src.id_data.constraints_extra
            #     # if src.name in con_list:
            #         # con_entry = con_list[src.name]
            #     # else:
            #         # con_entry = con_list.add()
            #         # con_entry.name = src.name

            #     # # Constraints entries
            #     # con_constraints = con_entry.constraints
            #     # con_item = con_constraints.add()
            #     # con_item.name = constraint.name

            #     # con_item.owner = owner_id
            #     # con_item.subowner = owner_bone
            #     # con_item.target = target_id
            #     # con_item.subtarget = target_bone

            #     # con_item.show_expanded = src is target
            # # else:
            #     # con_list = Constraint.get_dict(src.id_data, 'constraints_extra')
            #     # con_entry = Constraint.get_dict(con_list, src.name)

            #     # # Constraints entries
            #     # con_constraints = Constraint.get_dict(con_entry, 'constraints')

            #     # con_constraints[constraint.name] = dict(
            #         # owner=owner_id,
            #         # subowner=owner_bone,
            #         # target=target_id,
            #         # subtarget=target_bone,
            #         # show_expanded=src is target,
            #     # )

    # endregion
    ###############################################################################


class Constraint(Constraint):
    class new():
        """
        When called with arg(start=True), this will run like an operator\\
        The operation will generate the constraint type specified\\
        As well as run the related tasks\\
        Otherwise, must call the functions manually
        """
        def __init__(self, context, **kwargs):
            # Defaults
            self.show_expanded = True

            self.type = ''  # 'CAMERA_SOLVER'
            self.target = 'TO_ACTIVE'  # 'NONE'
            self.new_armature = True
            self.hash_name = False

            self.target_space = 'WORLD'
            self.owner_space = 'WORLD'
            self.use_local_location = True
            self.track_axis = 'TRACK_Y'
            self.track_offset = 0
            self.space_object = 'WORLD'

            self.at_tail = False
            self.add_drivers = False
            self.add_relation = False
            self.bake_animation = False
            self.bake_mode = 'range'
            self.display_bake = True
            # self.push_to_nla = False  # Only push if generated new objects
            self.copy_chain = False
            # self.copy_chain_list = dict()
            self.head_tail = 0.0
            self.armature_display_type = 'RANDOM'
            self.map_from = 'LOCATION'
            self.map_to = 'LOCATION'
            self.rotation_x = 0.0
            self.rotation_y = 0.0
            self.rotation_z = 0.0
            self.use_mesh = False
            self.do_physics = False

            self.is_custom = bool(self.type in Constraint.custom_types)
            self.selected = None

            self.relations = list()
            # self.relations = dict()
            self.start = False

            # Overwrite defaults with manual values
            for arg in kwargs:
                setattr(self, arg, kwargs[arg])

            if self.selected is None:
                self.selected = list()
                if context.mode == 'POSE':
                    self.selected.extend(context.selected_pose_bones)
                    if len(self.selected) <= 1 and (self.type not in Constraint.no_target):
                        # When one bone is selected, add the selected meshes/empties
                        for src in context.selected_objects:
                            if not Is.armature(src):
                                self.selected.append(src)
                else:
                    for src in Get.selected(context):
                        if Is.armature(src) and (self.type not in Constraint.no_target):
                            bones = list()
                            for bone in src.pose.bones:
                                if bone.bone.select and Is.in_visible_armature_layers(bone, src.data):
                                    bones.append(bone)
                            if bones:
                                self.selected.extend(bones)
                            else:
                                self.selected.append(src)
                        else:
                            self.selected.append(src)

            if self.start:
                self.init_objects_relations(context)
                self.add_constraints(context, self.relations)

        #####################################
        # Process

        def init_objects_relations(self, context, active=None, new_objects=None):
            """
            Create new bones/objects (if applicable)\\
            Then mark the owners + targets for later use in constraint creation\\
            new_objects is a dictionary containing
                items from selected and their target:
                new_objects[src] = new_control
            """

            if active is None:
                active = Get.active(context)

            if (self.target == 'TO_ACTIVE' and len(self.selected) == 1) or \
                (self.type in Constraint.curves and self.target not in (
                    'TO_ARMATURE', 'TO_OBJECT', 'NONE')
                ):
                if Is.posebone(self.selected[0]):
                    self.target = 'TO_ARMATURE'
                elif Is.object(self.selected[0]):
                    self.target = 'TO_OBJECT'
                else:
                    self.target = 'NONE'
                    # None SHOULD be the default

            # By default, use the bone's rest pose in edit, then update pose in pose mode
            self.use_local_transforms = True

            if (self.type == 'ARMATURE'):
                # Armature modifier starts offsetting from the rest pose
                self.use_local_transforms = False
            elif len({self.target_space, self.owner_space, 'LOCAL_WITH_PARENT', 'LOCAL'}) == 2:
                if self.type in ('COPY_LOCATION', 'COPY_ROTATION', 'COPY_SCALE', 'COPY_TRANSFORMS'):
                    # Use the original's final pose as the copy's default
                    self.use_local_transforms = False

            if (self.type == 'SHRINKWRAP'):
                if self.target == 'NONE':
                    pass
                elif self.target in ('TO_ARMATURE', 'TO_OBJECT'):
                    self.target = 'TO_OBJECT'
                else:
                    for src in self.selected:
                        if Is.mesh(src):
                            break
                    else:
                        self.target = 'TO_OBJECT'
                self.use_mesh = True

            if self.type in Constraint.no_target:
                new_objects = dict()
            elif new_objects is None:
                new_objects = self.generate_new_objects(context,
                    selected=self.selected)

            if self.bake_animation:
                # if no new bones/objects were created, then don't bake anything
                self.bake_animation = bool(new_objects)

            for src in self.selected:
                if self.target == 'NONE':
                    owner = src
                    target = None
                elif self.target == 'TO_ACTIVE':
                    if src == active: continue
                    owner = src
                    target = active
                elif self.target == 'FROM_ACTIVE':
                    if src == active: continue
                    owner = active
                    target = src
                elif self.target in ('TO_ARMATURE', 'TO_OBJECT'):
                    owner = src
                    target = new_objects.get(src)
                    if self.add_relation and target:
                        Constraint.add_original_relation(context, target, owner, self.use_local_transforms)
                elif self.target in ('FROM_ARMATURE', 'FROM_OBJECT'):
                    owner = new_objects.get(src)
                    target = src
                    if self.add_relation and owner:
                        Constraint.add_original_relation(context, owner, target, self.use_local_transforms)
                else:
                    assert None, self.target

                "This is used externally by the operator for the next step"
                self.relations.append((owner, target))

        def add_constraints(self, context, pairs):
            """
            create constraints for the items of (owner, target) in pairs\\
            Owner is the item holding the constraint\\
            Target is the target of the constraint
            """
            assert self.type, (
                self.type, "not valid type for inserting constraints")

            # self.find_prop_desc(kwargs, self.type)

            if self.type == 'SPLINE_IK':
                self.spline_ik(context, pairs)
            else:
                for (owner, target) in pairs:
                    self.add_constraint(context, owner, target)

        ########################################
        # Functions

        def add_constraint(self, context, owner, target):
            # self.find_prop_desc(kwargs, type)
            type = self.type

            limit = None
            limits = ('LIMIT_LOCATION', 'LIMIT_ROTATION', 'LIMIT_SCALE')
            if type not in limits:
                for limit_con in reversed(owner.constraints):
                    if limit_con.type in limits:
                        limit = limit_con
                    else:
                        break
            if limit:
                for (index, limit_con) in enumerate(owner.constraints):
                    if limit_con == limit:
                        iters = index
                        break
            else:
                index = iters = None

            def iterate(count):
                nonlocal index, iters
                if index is None:
                    index = iter(-1 for x in range(count))
                else:
                    index = iter(iters + x for x in range(count))

            if not Is.posebone(owner) and type in {'ik_transforms', 'SPLINE_IK', 'IK'}:
                return

            # Custom constraint setups
            if type == 'ik_transforms':
                iterate(3)

                con = self.new_constraint(owner, 'IK', next(index))
                con.use_tail = self.at_tail
                self.set_constraint(context, con, owner, target)
                if not self.at_tail:
                    con.chain_count -= 1  # IK setup is different, requiring 1 less length

                # mute if bone doesn't have parent, and it's not at the bone's tail
                con.mute = not bool((self.at_tail or owner.parent))

                con = self.new_constraint(owner, 'COPY_ROTATION', next(index))
                self.set_constraint(context, con, owner, target)
                con = self.new_constraint(owner, 'COPY_SCALE', next(index))
                self.set_constraint(context, con, owner, target)
            elif type == 'pivot':
                iterate(2)
                con = self.new_constraint(owner, 'COPY_LOCATION', next(index))
                self.set_constraint(context, con, owner, target)
                con.invert_x = con.invert_y = con.invert_z = True
                con.use_offset = True
                con.target_space = con.owner_space = 'LOCAL'
                loc = target.location.copy()
                target.location = (0, 0, 0)
                con = self.new_constraint(owner, 'CHILD_OF', next(index))
                self.set_constraint(context, con, owner, target)
                target.location = loc
            elif type == 'track_rotation':
                iterate(4)
                if Is.posebone(target) and not Is.connected(owner):
                    con = self.new_constraint(owner, 'COPY_LOCATION', next(index))
                    self.set_constraint(context, con, owner, target)
                    con.head_tail = 0
                con = self.new_constraint(owner, 'COPY_SCALE', next(index))
                self.set_constraint(context, con, owner, target)
                # mimic stretch constraints
                con.use_x = False
                con.use_z = False

                con = self.new_constraint(owner, 'COPY_ROTATION', next(index))
                self.set_constraint(context, con, owner, target)
                con = self.new_constraint(owner, 'DAMPED_TRACK', next(index))
                self.set_constraint(context, con, owner, target)
            elif type == 'stretch_rotation':
                iterate(4)
                if Is.posebone(target) and not Is.connected(owner):
                    con = self.new_constraint(owner, 'COPY_LOCATION', next(index))
                    self.set_constraint(context, con, owner, target)
                    con.head_tail = 0
                con = self.new_constraint(owner, 'COPY_SCALE', next(index))
                self.set_constraint(context, con, owner, target)
                # mimic stretch constraints
                con.use_x = False
                con.use_z = False

                con = self.new_constraint(owner, 'COPY_ROTATION', next(index))
                self.set_constraint(context, con, owner, target)
                con = self.new_constraint(owner, 'STRETCH_TO', next(index))
                self.set_constraint(context, con, owner, target)
            elif type.startswith('child_of_'):
                for t in ('location', 'rotation', 'scale'):
                    if type == f'child_of_{t}':
                        iterate(1)
                        con = self.new_constraint(owner, 'CHILD_OF', next(index))
                        self.set_child_of(
                            con, t == 'location', t == 'rotation', t == 'scale')
                        self.set_constraint(context, con, owner, target)
                        break
                else:
                    if type == 'child_of_locrot':
                        iterate(2)
                        con = self.new_constraint(owner, 'CHILD_OF', next(index))
                        self.set_child_of(con, True, False, False)
                        self.set_constraint(context, con, owner, target)
                        con = self.new_constraint(owner, 'CHILD_OF', next(index))
                        self.set_child_of(con, False, True, True)
                        self.set_constraint(context, con, owner, target)
                    elif type == 'child_of_locscale':
                        iterate(1)
                        con = self.new_constraint(owner, 'CHILD_OF', next(index))
                        self.set_child_of(con, True, False, True)
                        self.set_constraint(context, con, owner, target)
                    elif type == 'child_of_rotscale':
                        iterate(1)
                        con = self.new_constraint(owner, 'CHILD_OF', next(index))
                        self.set_child_of(con, False, True, True)
                        self.set_constraint(context, con, owner, target)
            elif type == 'offset_transforms':
                # for space in ('LOCATION', 'ROTATION', 'SCALE'):
                    iterate(1)
                    con = self.new_constraint(owner, 'TRANSFORM', next(index))
                    self.set_constraint(context, con, owner, target)
                    # con.map_from = con.map_to = space
                    con.map_from = self.map_from
                    con.map_to = self.map_to
                    con.target_space = con.owner_space = 'LOCAL'

            # Default constraints
            elif type == 'SPLINE_IK':
                """Queue Spline, don't add constraints yet"""
                iterate(1)
                # self.ik_splines[owner] = target
                # con = self.new_constraint(owner, type, next(index))
                con = None
            elif type == 'ARMATURE':
                for con in owner.constraints:
                    if con.type == 'ARMATURE':
                        # Reuse existing constraint
                        break
                else:
                    # Create a new constraint
                    iterate(1)
                    con = self.new_constraint(owner, type, next(index))
                self.set_constraint(context, con, owner, target)
            else:
                iterate(1)
                con = self.new_constraint(owner, type, next(index))
                self.set_constraint(context, con, owner, target)

            if self.do_physics:
                self.generate_physic_objects(context, owner, target)

            return con

        def add_constraint_driver(self, con, target, prop=None, desc=None):
            """
            Add custom prop to target and use it as a driver on the constraint
            """

            info = Constraint.custom_types.get(self.type)
            is_custom = bool(info)
            if info is None: info = Constraint.constraint_types[self.type]
            (prop_, desc_) = info
            # if is_custom: prop_ += ' - ' + Constraint.constraint_types[con.type][0]

            if desc is None: desc = desc_
            if prop is None:
                # Use the Constraint Label and a unique number (for this run)
                prop = prop_ + '||' + str(hash(target)) + '||' + str(hash(self))

            label = prop_  # The display text for Constraints Custom Prop UI
            # prop_obj = prop + '||' + str(hash(target))  # Unique id per control

            # if prop in target:
                # name = prop
                # num = 1
                # while prop in target:
                #     # prop = f'{name}.{num:03}'  # .001 incremental padding
                #     prop = f'{name}{num}'  # custom props don't use padding
                #     num += 1
                # # Add incremental number until not duplicate

            # Add an object level reference to the custom prop
            for src in (con, target):
                if (src == target) and (target.id_data == src.id_data):
                    continue
                src = src.id_data

                if hasattr(src, 'constraints_props'):
                    entry_props = src.id_data.constraints_props.props

                    entry = entry_props.get(prop)
                    # print('entry:', getattr(entry, 'name', "No Entry"), 'prop_obj:', prop_obj)
                    # if entry:
                    #     print('target:', target.name)
                    #     print('Subtar:', entry.subtarget)
                    #     print('-------------')
                    #     if target.name == entry.subtarget:
                    #         entry = None
                    # # while entry:
                    # #     if entry.target == target.id_data.name:
                    # #         if Is.posebone(target):
                    # #             if entry.subtarget == target.name:
                    # #                 break
                    # #         else:
                    # #             break
                    # #     entry = None
                    if entry is None:
                        # Don't create duplicates
                        entry = entry_props.add()
                        entry.name = prop
                else:
                    con_props = Constraint.get_dict(src, 'constraints_props')
                    con_props['show_expanded'] = con_props.get('show_expanded', 1)
                    entry_props = Constraint.get_dict(con_props, 'props')

                    entry = entry_props.get(prop)
                    # print('entry:', getattr(entry, 'name', "No Entry"), 'prop:', prop)
                    # if entry:
                    #     print('target:', target.name)
                    #     print('Subtar:', entry.subtarget)
                    #     print('-------------')
                    #     if target.name == entry.subtarget:
                    #         entry = None
                    # # while entry:
                    # #     if entry['target'] == target.id_data.name:
                    # #         if Is.posebone(target):
                    # #             if entry['subtarget'] == target.name:
                    # #                 break
                    # #         else:
                    # #             break
                    # #     entry = None
                    if entry is None:
                        entry_props[prop_obj] = dict(prop='', target='', subtarget='')
                        entry = entry_props[prop]

                entry['target'] = target.id_data.name
                if Is.posebone(target):
                    entry['subtarget'] = target.name
                entry['prop'] = label

            if not target.get(prop):
                # Don't reset it if already used
                target[prop] = 1.0
                target['_RNA_UI'] = target.get('_RNA_UI', dict())
                target['_RNA_UI'][prop] = dict(
                    min=0.0, max=1.0,
                    soft_min=0.0, soft_max=1.0,
                    description=desc,
                    default=0.0,
                )

            driver_ = con.driver_add('influence')
            driver = driver_.driver
            # driver.expression = 'var'
            driver.type = 'AVERAGE'
            variable = driver.variables.new().targets[0]
            variable.id_type = 'OBJECT'
            variable.id = target.id_data

            variable.data_path = target.path_from_id(f'["{prop}"]')

            driver.expression += " "
            driver.expression = driver.expression[:-1]

            return driver_

        def find_prop_desc(self, kwargs, type):
            """
            If using add_drivers, search through constraint list one time\\
            to find the property and description for the constraint type.\\
            (if 'prop' or 'desc' not already defined)
            """

            if kwargs.get('add_drivers', getattr(self, 'add_drivers', False)):
                prop = kwargs.get('prop')
                desc = kwargs.get('desc')

                if None in (prop, desc):
                    for cti in Constraint.constraint_type_items:
                        if cti[0] == type:
                            if prop is None:
                                # prop = f'{cti[1]}||{hash(self)}'
                                prop = cti[1]
                            if desc is None:
                                desc = cti[2]
                            break
                    else:
                        assert None, ("Constraint has type but it's not labeled")
                    kwargs['prop'] = prop
                    kwargs['desc'] = desc

            return kwargs

        def generate_new_objects(self, context, selected=list()):
            """
            Generate new objects/armatures (in relation to each item in selected)\\
            Then link the new items to the old;  (objects[src] = new_src)\\
            Finally return this relation as a dictionary
            """

            first_active = Get.active(context)
            active = None

            new_arm = self.target in ('TO_ARMATURE', 'FROM_ARMATURE')
            new_obj = self.target in ('TO_OBJECT', 'FROM_OBJECT')
            new_bones = new_arm and not self.new_armature
            new_objects = dict()

            if not (new_arm or new_obj or new_bones):
                return new_objects
            # else:
                # # New objects generated, so allow push to nla
                # self.push_to_nla = True


            # Don't use connect on bones for these constraints
            is_not_tracker = self.type not in (
                *Constraint.curves,
                'DAMPED_TRACK', 'IK', 'LOCKED_TRACK',
                'STRETCH_TO', 'TRACK_TO',
                # Custom
                'ik_transforms', 'track_rotation', 'stretch_rotation', 'look_at',
            )

            if context.scene.use_armature_target:
                armature = context.scene.armature_target
                if armature and (context.scene not in armature.users_scene):
                    armature = context.scene.armature_target = None
            else:
                armature = None

            scn_armature = False
            if (new_arm and armature):
                scn_armature = str(hash(armature))
                self.new_armature = False
                new_bones = False
                Set.mode(context, armature, 'EDIT')
            elif (new_arm and not new_bones):
                armature = New.armature(context,
                    display_type=self.armature_display_type)
                armature.name = utils.string(
                    self.type.replace('_', ' ').title(), '||', hash(armature)
                )

                Set.object_display_type(armature, 'WIRE')
                Set.xray(armature, True)

                Set.mode(context, armature, 'EDIT')
            else:
                armature = None

            for src in selected:
                if new_obj:
                    name = utils.string(
                        self.type.replace('_', ' ').title(),
                        '||', src.id_data.name.split('||', 1)[0],
                        '||', src.name,
                        # '||', hash(src),
                    )

                    if self.use_mesh:
                        obj = New.mesh(context, name=name)
                    else:
                        obj = New.object(context, name=name)

                        if Is.posebone(src):
                            width = src.bone.bbone_x
                            depth = src.bone.bbone_z
                            height = src.length
                            size = sum((width, depth, height))
                        elif Is.empty(src):
                            size = 1.0
                        else:
                            size = sum(src.dimensions)
                        Set.empty_size(obj, size)

                    self.init_nla(src.id_data, obj)
                    new_objects[src] = obj
                    self.set_object(context, obj, src)
                    if (context.mode == 'OBJECT'):
                        if (active is None or first_active == src):
                            Set.active(context, obj)
                            Set.select(obj, True)
                            Set.select(src, False)
                            active = obj
                elif new_arm or new_bones:
                    if new_bones:
                        if not Is.posebone(src):
                            continue
                        armature = src.id_data
                        name = utils.string(
                            # self.type.replace('_', ' ').title(), '||',
                            src.name,
                        )
                        if armature.mode != 'EDIT':
                            Set.mode(context, armature, 'EDIT')
                    else:
                        name = utils.string(
                            # self.type.replace('_', ' ').title(), '||',
                            src.name,
                        )

                    if scn_armature:
                        # Using preset object which may not be the "same" armature
                        name = utils.string(
                            src.id_data.name.split('||', 1)[0],
                            '||', self.type.replace('_', ' ').title(),
                            # '||', scn_armature,
                            '||', src.name,
                            # '||', hash(src),
                        )
                    elif not(self.new_armature):
                        # Since bone is added in same armature, it needs unique name
                        if name.lower().endswith(('.l', '.r')):
                            if (self.type == 'ik_transforms'):
                                name = name[:-2] + '.IK' + name[-2:]
                            else:
                                name = name[:-2] + '.' + self.type + name[-2:]
                        else:
                            name = (name + '.' + self.type)

                    self.init_nla(src.id_data, armature)
                    bone = New.bone(context, armature, name=name)
                    self.set_edit_bone(context, bone, src)
                    new_objects[src] = (armature, bone.name)

                    if self.create_widgets and (src.id_data.data != bone.id_data):
                        # Add display bone
                        vis = New.bone(context, armature, name='VIS-' + bone.name)
                        self.set_edit_bone(context, vis, src)

            if new_arm or new_bones:
                armatures = set()  # objects to set back to pose later

                for src in new_objects:
                    (armature, name) = new_objects[src]
                    armatures.add(armature)
                    if self.copy_chain:
                        # self.copy_chain_list[src] =
                        if not src.parent:
                            continue

                        bone = armature.data.edit_bones[name]
                        # mat = Get.matrix(bone)
                        for psrc in src.parent_recursive:
                            if psrc not in new_objects:
                                continue
                            (armature, pname) = new_objects[psrc]
                            pbone = armature.data.edit_bones[pname]
                            bone.parent = pbone
                            if (psrc == src.parent) and (is_not_tracker):
                                bone.use_connect = src.bone.use_connect
                            break

                for o in armatures:
                    Set.select(armature, True)
                    Set.mode(context, armature, 'POSE')

                ordered_objects = list()
                for src in new_objects:
                    if src.parent in ordered_objects:
                        ordered_objects.insert(ordered_objects.index(src.parent) + 1, src)
                    else:
                        for child in src.children:
                            if child in ordered_objects:
                                ordered_objects.insert(ordered_objects.index(child) + 1, src)
                                break
                        else:
                            ordered_objects.append(src)

                for src in reversed(ordered_objects):
                    (armature, name) = new_objects[src]
                    bone = armature.pose.bones[name]
                    self.set_pose_bone(context, bone, src)
                    Set.bone_group(bone, f"{self.type}||{hash(self)}", color=True)
                    new_objects[src] = bone
                    Set.select(src, False)
                    Set.select(bone, True)
                    if (active is None or first_active == src):
                        Set.active(context, bone)
                        active = bone

            return new_objects

        def generate_physic_objects(self, context, owner, target):
            """Add two mesh objects and an empty, with physics preset"""
            # """Add constraint to target, containing physics meshes"""

            name = utils.string(
                "Physics",
                '||', target.id_data.name.split('||', 1)[0],
                '||', target.name,
            )
            if Is.posebone(target):
                width = target.bone.bbone_x
                height = target.length / 2
                depth = target.bone.bbone_z
                size = sum((width, depth, height))
            elif Is.empty(target):
                (width, depth, height) = (1, 1, 1)
                size = 1
            else:
                (width, depth, height) = target.dimensions
                size = sum((width, depth, height))

            shape = dict(width=width, depth=depth, height=height)
            if Is.posebone(target):
                shape['loc'] = (0, 0, 1)
                shape['rot'] = (-90,)

            mesh1 = New.mesh(context, name=name + '-Collision', type='CUBE', **shape)
            mesh2 = New.mesh(context, name=name + '-Bind', type='POINT')
            pivot = New.object(context, name=name + '-Pivot')

            # Move the meshes to target
            # mesh1.parent = mesh2.parent = pivot
            mesh1.matrix_world = Get.matrix(target)
            # mesh2.matrix_world = pivot.matrix_world = Get.matrix(target, tail=True)
            mesh2.matrix_world = Get.matrix(target)
            pivot.matrix_world = Get.matrix(target, tail=True)

            # Change the look default look of the meshes
            Set.object_display_type(mesh1, 'WIRE')
            Set.object_display_type(mesh2, 'WIRE')
            Set.empty_size(pivot, size)

            # Add physics to the meshes
            bpy.ops.rigidbody.object_add(dict(object=mesh1))
            bpy.ops.rigidbody.object_add(dict(object=mesh2))
            active = Get.active(context)
            Set.active(context, pivot)
            bpy.ops.rigidbody.constraint_add()
            Set.active(context, active)

            # Set default physics parameters
            mesh1.rigid_body.collision_shape = 'MESH'
            mesh2.rigid_body.kinematic = True
            pivot.rigid_body_constraint.object1 = mesh1
            pivot.rigid_body_constraint.object2 = mesh2

            con = self.new_constraint(pivot, 'CHILD_OF')
            self.set_constraint(context, con, pivot, target)
            con = self.new_constraint(mesh1, 'COPY_TRANSFORMS')
            self.set_constraint(context, con, mesh1, target)
            con = self.new_constraint(mesh2, 'CHILD_OF')
            self.set_constraint(context, con, mesh2, pivot)

            # Add a constraint to lock the original bone/object to the collision mesh
            con = self.new_constraint(owner, 'COPY_TRANSFORMS')
            con.name = 'Physics || Copy Transforms'
            self.set_constraint(context, con, owner, mesh1)

        def new_constraint(self, owner, type, index=-1):
            """Add a constraint to owner, then name and return it"""

            active_index = len(owner.constraints)
            constraint = owner.constraints.new(type)
            if index >= 0:
                owner.constraints.move(active_index, index)

            if self.hash_name:
                constraint.name = f"{type.title()}-{hash(constraint)}"
            constraint.show_expanded = self.show_expanded

            return constraint

        def set_child_of(self, constraint, loc, rot, scale):
            """macro to toggle the loc/rot/scale for a child_of constraint"""
            for v in ('x', 'y', 'z'):
                setattr(constraint, f'use_location_{v}', loc)
                setattr(constraint, f'use_rotation_{v}', rot)
                setattr(constraint, f'use_scale_{v}', scale)

        def set_constraint(self, context, con, owner, target=None):
            """
            Set the default values of the constraint\\
            Also perform "fixes" if needed
            """

            owner_type = ('OBJECT', 'BONE')[Is.posebone(owner)]

            def scan_ik(owner, disable_ik=dict()):
                """log IK chain mute statuses then mute them"""
                if owner.is_in_ik_chain:
                    for bc in [owner] + owner.children_recursive:
                        for bc_con in bc.constraints:
                            if bc_con.type in {'IK', 'SPLINE_IK'}:
                                disable_ik[bc, bc_con] = bc_con.mute
                                bc_con.mute = True
                return disable_ik

            if target:
                fix_inverse = False
                if con.type == 'CHILD_OF':
                    fix_inverse = False

                if fix_inverse:
                    # Remember pose to reset inverse
                    disable_ik = dict()
                    if Is.posebone(owner):
                        if owner.is_in_ik_chain:
                            disable_ik = scan_ik(owner)
                        else:
                            for b in owner.children:
                                disable_ik = scan_ik(b, disable_ik)
                    if disable_ik:
                        utils.update(context)
                    original_matrix = Get.matrix(owner)

                if Is.posebone(target):
                    subtarget = target.name
                else:
                    subtarget = ""

                if hasattr(con, 'target'):
                    con.target = target.id_data
                    if hasattr(con, 'subtarget') and subtarget:
                        con.subtarget = subtarget
                elif hasattr(con, 'targets'):
                    # Armature constraint
                    sub = con.targets.new()
                    sub.target = target.id_data
                    if subtarget:
                        sub.subtarget = subtarget

                if fix_inverse:
                    utils.update(context)
                    reset_matrix = Get.matrix(owner)
                    con.inverse_matrix = utils.multiply_matrix(
                        original_matrix, reset_matrix.inverted()
                    )

                    # try:
                    # except:
                    # if (con.use_location_x and not con.use_rotation_x):
                    #     cc = context.copy()
                    #     cc['constraint'] = con
                    #     cc['object'] = owner.id_data
                    #     bpy.ops.constraint.childof_set_inverse(
                    #         cc, constraint=con.name, owner=owner_type)

                    # unmute muted ik chains
                    for (bc, bc_con) in disable_ik:
                        bc_con.mute = disable_ik[bc, bc_con]

            if hasattr(con, 'target_space'):
                tspace = self.target_space
                if self.target_space != 'WORLD':
                    if not Is.posebone(target):
                        tspace = 'LOCAL'
                    if self.type == 'ARMATURE':
                        # Armature only supports ('WORLD', 'LOCAL')
                        # However Local is not the result I want
                        tspace = 'WORLD'
                con.target_space = tspace

            if hasattr(con, 'owner_space'):
                ospace = self.owner_space
                if self.owner_space != 'WORLD':
                    if not Is.posebone(owner):
                        ospace = 'LOCAL'
                    if self.type == 'ARMATURE':
                        # Armature only supports ('WORLD', 'LOCAL')
                        # However Local is not the result I want
                        ospace = 'WORLD'
                con.owner_space = ospace

            if hasattr(con, 'head_tail'):
                con.head_tail = self.head_tail
                if (self.type in {'track_rotation', 'stretch_rotation'} and
                        not (self.at_tail or self.head_tail)) or \
                    (self.type == 'STRETCH_TO' and
                        self.target in ('TO_ARMATURE', 'TO_OBJECT')):
                    # By default, instead of forcing the bone to the tail, use its tail
                    con.head_tail = 1.0

            if hasattr(con, 'track_axis'):
                con.track_axis = Constraint.get_track_axis(owner, self.track_axis)

            if hasattr(con, 'space_object'):
                con.space_object = self.space_object

            if len({self.target_space, self.owner_space, 'LOCAL_WITH_PARENT', 'LOCAL'}) == 2:
                if con.type == 'COPY_TRANSFORMS':
                    con.mix_mode = 'AFTER'
                    # mix_mode_items = [
                        # ('REPLACE', "Replace", "Replace the original transformation with copied"),
                        # ('BEFORE', "Before Original", "Apply copied transformation before original, as if the constraint target is a parent. "
                                                    #   "Scale is handled specially to avoid creating shear"),
                        # ('AFTER', "After Original", "Apply copied transformation after original, as if the constraint target is a child. "
                                                    # "Scale is handled specially to avoid creating shear"),
                    # ]
                elif con.type == 'COPY_LOCATION':
                    con.use_offset = True
                elif con.type == 'COPY_ROTATION':
                    con.mix_mode = 'AFTER'
                    # mix_mode_items = [
                        # ('REPLACE', "Replace", "Replace the original rotation with copied"),
                        # ('ADD', "Add", "Add euler component values together"),
                        # ('BEFORE', "Before Original", "Apply copied rotation before original, as if the constraint target is a parent"),
                        # ('AFTER', "After Original", "Apply copied rotation after original, as if the constraint target is a child"),
                        # ('OFFSET', "Offset (Legacy)", "Combine rotations like the original Offset checkbox. Does not work well for " "multiple axis rotations"),
                    # ]
                elif con.type == 'COPY_SCALE':
                    con.use_offset = True

            if con.type == 'ARMATURE':
                con.use_deform_preserve_volume = True

            elif con.type == 'IK':
                con.chain_count = 1  # Fixes length for default IK constraint
                prev = owner
                for b in owner.parent_recursive:
                    if Is.connected(prev, pseudo=not owner.bone.use_connect):
                        con.chain_count += 1
                        prev = b
                    else:
                        break

            elif con.type == 'FLOOR':
                # I could use the track_axis property or a separate property
                # but use a default instead
                con.floor_location = 'FLOOR_NEGATIVE_Z'
                con.use_rotation = True

            elif con.type == 'LIMIT_DISTANCE':
                con.limit_mode = 'LIMITDIST_ONSURFACE'
                active = Set.active(context, owner)
                bpy.ops.constraint.limitdistance_reset(
                    dict(constraint=con), constraint=con.name, owner=owner_type)
                Set.active(context, active)
            elif con.type == 'LIMIT_LOCATION':
                for xyz in 'xyz':
                    if getattr(owner.location, xyz) < 0:
                        setattr(con, 'min_' + xyz, getattr(owner.location, xyz))
                        setattr(con, 'max_' + xyz, 0)
                    else:
                        setattr(con, 'min_' + xyz, 0)
                        setattr(con, 'max_' + xyz, getattr(owner.location, xyz))
                con.use_min_x = con.use_min_y = con.use_min_z = True
                con.use_max_x = con.use_max_y = con.use_max_z = True
                con.owner_space = 'LOCAL'
            elif con.type == 'LIMIT_ROTATION':
                # con.min_x = con.min_y = con.min_z = radians(-360)
                # con.max_x = con.max_y = con.max_z = radians(360)
                if owner.rotation_mode == 'QUATERNION':
                    rot = owner.rotation_quaternion.to_euler('XYZ')
                elif owner.rotation_mode == 'AXIS_ANGLE':
                    rot = owner.rotation_quaternion.to_euler('XYZ')
                    # I don't know how to convert axis_angle to anything else, consistently
                else:
                    rot = owner.rotation_euler
                for xyz in 'xyz':
                    if getattr(rot, xyz) < 0:
                        setattr(con, 'min_' + xyz, getattr(rot, xyz))
                        setattr(con, 'max_' + xyz, 0)
                    else:
                        setattr(con, 'min_' + xyz, 0)
                        setattr(con, 'max_' + xyz, getattr(rot, xyz))
                con.use_limit_x = con.use_limit_y = con.use_limit_z = True
                con.owner_space = 'LOCAL'
            elif con.type == 'LIMIT_SCALE':
                for xyz in 'xyz':
                    if getattr(owner.scale, xyz) < 1:
                        setattr(con, 'min_' + xyz, getattr(owner.scale, xyz))
                        setattr(con, 'max_' + xyz, 1)
                    else:
                        setattr(con, 'min_' + xyz, 1)
                        setattr(con, 'max_' + xyz, getattr(owner.scale, xyz))
                con.use_min_x = con.use_min_y = con.use_min_z = True
                con.use_max_x = con.use_max_y = con.use_max_z = True
                con.owner_space = 'LOCAL'

            elif con.type == 'SHRINKWRAP':
                con.wrap_mode = 'OUTSIDE'

            elif con.type == 'STRETCH_TO':
                con.keep_axis = 'SWING_Y'

            # I honestly don't know how to use the transform constraint :p
            elif con.type == 'TRANSFORM':
                con.map_from = self.map_from
                con.map_to = self.map_to
                con.use_motion_extrapolate = True
                for dest in ('from', 'to'):
                    for axis in ('x', 'y', 'z'):
                        for p, v in {
                                ('', 10),
                                ('_rot', radians(10)),  # 180
                                ('_scale', 10)}:
                            setattr(con, f'{dest}_min_{axis}{p}', -v)
                            setattr(con, f'{dest}_max_{axis}{p}', v)
            # elif type == 'TRANSFORM':
                # spaces = self.map_from_to
                # if not spaces:
                    # spaces = {'LOCATION'}
                # for space in ('LOCATION', 'ROTATION', 'SCALE'):
                    # if space not in spaces:
                        # continue
                    # con = self.new_constraint(owner, type)
                    # self.set_constraint(context, con, owner, target)
                    # con = self.new(owner, type)
                    # con.show_expanded = expand_constraints
                    # con.map_from = con.map_to = space
                    # self.set(con, owner, target)
                    # self.fix(con, owner, target, is_custom)
                    # self.add_extra(owner, target, con)

            # if self.add_relation:
                # Constraint.add_constraint_relation(context, con, owner, target)

            if self.add_drivers and hasattr(con, 'influence') and target:
                prop = None  # kwargs.get('prop')
                desc = None  # kwargs.get('desc')
                self.add_constraint_driver(con, target, prop, desc)

        def set_edit_bone(self, context, bone, src):
            """Set the default pose for newly created bones"""

            from math import degrees
            from mathutils import Vector

            # When keyframing, use world space or local space
            if (self.type == 'pivot') and Is.posebone(src):
                bone.use_local_location = src.bone.use_local_location
            else:
                bone.use_local_location = self.use_local_location

            # Set the bone's default transforms
            bone.tail = Vector((0, 0, 1))
            bone.matrix = Get.matrix(src, local=self.use_local_transforms)

            if Is.posebone(src):
                for prop in ('length', 'bbone_x', 'bbone_z'):
                    setattr(bone, prop, getattr(src.bone, prop))

                bone.use_inherit_rotation = src.bone.use_inherit_rotation
                bone.inherit_scale = src.bone.inherit_scale

            bone.use_deform = False

            # if self.at_tail:
                # bone.translate(bone.tail - bone.head)
                # # bone.matrix.translation = bone.matrix.Translation(
                # #     bone.tail - bone.head).translation
            # elif self.head_tail:
                # slide = (bone.tail - bone.head)
                # tran = Vector(lerp(bone.head, slide, self.head_tail))
                # bone.translate(tran)

            # if self.type in {'DAMPED_TRACK'}:
                # head_tail = 0
            # elif self.at_tail or self.type in {'STRETCH_TO'}:
                # head_tail = 1
            # else:
                # head_tail = self.head_tail
            head_tail = self.at_tail

            if head_tail:
                # tail = multiply_matrix(
                #     Vector((0, bone.length, 0)),
                #     bone.matrix.inverted(),
                # )
                slide = Vector(utils.lerp(bone.head, bone.tail, head_tail))
                bone.translate(slide)
                # b.location += multipy_matrix(Vector((0, b.length, 0)) @ b.matrix_basis.inverted()

            # if self.copy_chain and src.parent in self.copy_chain_list:
                # bone.parent = bone.id_data.edit_bones[
                #     self.copy_chain_list[src.parent].name]
                # bone.use_connect = src.bone.use_connect
            if (self.rotation_x or self.rotation_y or self.rotation_z):
                new_matrix = utils.rotate_matrix(bone.matrix, (
                    *[degrees(getattr(self, f'rotation_{_}'))
                        for _ in ('x', 'y', 'z')],
                ))
                # nonlocal difference
                # difference = bone.matrix - new_matrix
                bone.matrix = new_matrix

        def set_pose_bone(self, context, bone, src):
            """Set the visual pose for newly created bones"""

            self.set_locks(bone)

            # Switch back to Pose mode and transfer pose
            bone.rotation_mode = src.rotation_mode
            if self.use_local_transforms:
                # Bone should NOT be a Default 0 space
                bone.matrix = Get.matrix(src)
            if self.copy_chain:
                utils.update(context)
            # bone.rotation_quaternion = (bone.matrix + difference).to_quaternion()

            if (src.id_data.data == bone.id_data):
                # bone created in active armature
                bone.custom_shape_transform = src  # default widgets to the original
            elif self.create_widgets:
                vis = bone.id_data.pose.bones.get('VIS-' + bone.name)
                if vis:
                    self.set_widget(bone, src, vis)

            # if self.at_tail:
                # update()
                # bone.matrix.translation = bone.matrix.Translation(
                #     getattr(src, 'tail', bone.tail)
                #     ).translation
            # elif self.head_tail:
                # update()
                # tran = Vector(lerp(src.head, src.tail, self.head_tail))
                # bone.matrix.translation = bone.matrix.Translation(
                #     tran
                #     ).translation

            # from math import degrees
            # # # # # new_matrix = rotate_matrix(bone.matrix, (
            # # # # #     *[degrees(getattr(self, f'rotation_{_}'))
            # # # # #         for _ in ('x', 'y', 'z')],
            # # # # # ))
            # # # # # bone.matrix = new_matrix

            # if self.type in {'DAMPED_TRACK'}:
                # head_tail = 0
            # elif self.type in {'STRETCH_TO'} and not (self.at_tail or self.head_tail):
                # # By default, instead of forcing the bone to the tail, use its tail
                # head_tail = 0
            # elif self.at_tail:
                # head_tail = 1
            # else:
                # head_tail = self.head_tail
            head_tail = self.at_tail

            if head_tail:
                utils.update(context)

                if bone.bone.use_local_location:
                    tail = bone.location + utils.multiply_matrix(
                        Vector((0, bone.length, 0)),
                        bone.matrix_basis.inverted(),
                    )
                    bone.location = Vector(utils.lerp(bone.location, tail, head_tail))
                else:
                    Set.matrix(bone, Get.matrix(bone, tail=True))

            track_axis = 'None'

            if self.type in ('DAMPED_TRACK', 'track_rotation'):
                track_axis = self.track_axis
                # rm = bone.rotation_mode
                # bone.rotation_mode = 'XYZ'

                if self.track_axis == 'TRACK_X':
                    # bone.rotation_euler.y += radians(90)
                    angle = -90
                    axis = 'Z'
                elif self.track_axis == 'TRACK_Y':
                    # bone.rotation_euler.x += radians(0)
                    angle = 0
                    axis = 'X'
                elif self.track_axis == 'TRACK_Z':
                    # bone.rotation_euler.x += radians(90)
                    angle = 90
                    axis = 'X'
                elif self.track_axis == 'TRACK_NEGATIVE_X':
                    # bone.rotation_euler.y += radians(-90)
                    angle = 90
                    axis = 'Z'
                elif self.track_axis == 'TRACK_NEGATIVE_Y':
                    # bone.rotation_euler.x += radians(180)
                    angle = 180
                    axis = 'X'
                elif self.track_axis == 'TRACK_NEGATIVE_Z':
                    # bone.rotation_euler.x += radians(-90)
                    angle = -90
                    axis = 'X'
                else:  # Impossible
                    angle = 0
                    axis = 'X'

                utils.update(context)
                mat_rotated = utils.multiply_matrix(
                    Get.matrix(bone),
                    Matrix.Rotation(radians(angle), 4, axis),
                    Matrix.Translation(Vector((0, bone.length * self.track_offset / 100, 0))),
                )
                Set.matrix(bone, mat_rotated)
                utils.update(context)
                Set.matrix(bone, Get.matrix(bone, tail=True))
                # bone.rotation_mode = rm
            elif self.type in ('STRETCH_TO', 'stretch_rotation') and self.track_offset:
                utils.update(context)
                mat_rotated = utils.multiply_matrix(
                    Get.matrix(bone),
                    Matrix.Translation(Vector((0, bone.length * self.track_offset / 100, 0)))
                )
                Set.matrix(bone, mat_rotated)
            elif self.type == 'IK':
                utils.update(context)
                Set.matrix(bone, Get.matrix(bone, tail=True))

            # Store the base matrix
            if hasattr(bone, 'base_transforms'):
                base = bone.base_transforms
                base.location = bone.location
                base.rotation_axis_angle = bone.rotation_axis_angle
                base.rotation_euler = bone.rotation_euler
                base.rotation_quaternion = bone.rotation_quaternion
                base.scale = bone.scale
                base.track_axis = track_axis
                base.track_offset = self.track_offset
            else:
                bone['base_transforms'] = dict(
                    location=bone.location,
                    rotation_axis_angle=bone.rotation_axis_angle,
                    rotation_euler=bone.rotation_euler,
                    rotation_quaternion=bone.rotation_quaternion,
                    scale=bone.scale,
                    track_axis=track_axis,
                    track_offset=self.track_offset,
                )

        def set_object(self, context, obj, src):
            """Set the visual pose for newly created objects"""

            self.set_locks(obj)

            # Switch back to Pose mode and transfer pose
            obj.rotation_mode = src.rotation_mode
            obj.matrix_world = Get.matrix(src)

            # if self.type in {'DAMPED_TRACK'}:
                # head_tail = 0
            if self.at_tail or self.type in {'STRETCH_TO', 'stretch_rotation'}:
                head_tail = 1
            else:
                head_tail = self.head_tail

            if head_tail:
                utils.update(context)
                Set.matrix(obj, Get.matrix(src, tail=True))

            track_axis = 'None'

            if self.type in ('DAMPED_TRACK', 'track_rotation'):
                track_axis = self.track_axis

                if self.track_axis == 'TRACK_X':
                    angle = -90
                    axis = 'Z'
                elif self.track_axis == 'TRACK_Y':
                    angle = 0
                    axis = 'X'
                elif self.track_axis == 'TRACK_Z':
                    angle = 90
                    axis = 'X'
                elif self.track_axis == 'TRACK_NEGATIVE_X':
                    angle = 90
                    axis = 'Z'
                elif self.track_axis == 'TRACK_NEGATIVE_Y':
                    angle = 180
                    axis = 'X'
                elif self.track_axis == 'TRACK_NEGATIVE_Z':
                    angle = -90
                    axis = 'X'
                    obj.rotation_euler.x += radians(-90)
                else:  # Impossible
                    angle = 0
                    axis = 'X'

                utils.update(context)
                mat_rotated = utils.multiply_matrix(
                    Get.matrix(src),
                    Matrix.Rotation(radians(angle), 4, axis),
                )
                Set.matrix(obj, mat_rotated)
                utils.update(context)

                y = (src.length if Is.posebone(src) else 1)
                Set.matrix(obj, utils.multiply_matrix(
                    Get.matrix(src, tail=True),
                    Matrix.Translation(Vector((0, y * self.track_offset / 100, 0))),
                ))
            elif self.type in ('STRETCH_TO', 'stretch_rotation') and self.track_offset:
                utils.update(context)
                y = (src.length if Is.posebone(src) else 1)
                Set.matrix(obj, utils.multiply_matrix(
                    Get.matrix(src, tail=True),
                    Matrix.Translation(Vector((0, y * self.track_offset / 100, 0))),
                ))
            elif self.type == 'IK':
                utils.update(context)
                Set.matrix(obj, Get.matrix(src, tail=True))

            # Store the base matrix
            if hasattr(obj, 'base_transforms'):
                base = obj.base_transforms
                base.location = obj.location
                base.rotation_axis_angle = obj.rotation_axis_angle
                base.rotation_euler = obj.rotation_euler
                base.rotation_quaternion = obj.rotation_quaternion
                base.scale = obj.scale
                base.track_axis = track_axis
                base.track_offset = self.track_offset
            else:
                obj['base_transforms'] = dict(
                    location=obj.location,
                    rotation_axis_angle=obj.rotation_axis_angle,
                    rotation_euler=obj.rotation_euler,
                    rotation_quaternion=obj.rotation_quaternion,
                    scale=obj.scale,
                    track_axis=track_axis,
                    track_offset=self.track_offset,
                )

        def set_locks(self, src):
            """When creating new constraint controllers, lock unneeded attributes"""
            type = self.type

            location = rotation = scale = False

            if type in ('COPY_LOCATION', 'IK', 'child_of_location'):
                rotation = True
                scale = True
            elif type in ('COPY_ROTATION', 'child_of_rotation'):
                location = True
                scale = True
            elif type in ('COPY_SCALE', 'child_of_scale'):
                location = True
                rotation = True
            elif type in ('LIMIT_DISTANCE', 'DAMPED_TRACK', 'LOCKED_TRACK', 'TRACK_TO', 'STRETCH_TO', 'PIVOT'):
                if not (self.head_tail or self.at_tail):
                    rotation = True
                    scale = True
            elif type == 'FLOOR':
                scale = True
            else:
                pass
                # 'COPY_TRANSFORMS', 'TRANSFORM',
                # 'ACTION', 'ARMATURE', 'CHILD_OF', 'child_of_locrot', 'ik_transforms', 'pivot',
                # 'track_rotation', 'stretch_rotation'
                # *Constraint.curves, 'SHRINKWRAP',

            if location:
                src.lock_location[:] = (True, True, True)
            if rotation:
                src.lock_rotation[:] = (True, True, True)
                src.lock_rotation_w = True
            if scale:
                src.lock_scale[:] = (True, True, True)

        def set_widget(self, bone, src, vis):
            """Create Widgets for new bones"""

            wgt_op = utils.find_op('bonewidget.create_widget')

            def wgt(bone, **args):
                """
                args:
                    slide=(0.0, 0.0, 0.0)
                    rotate=(0.0, 0.0, 0.0)
                    global_size=1.0
                    scale=(1.0, 1.0, 1.0)
                """

                if not args.get('widget'):
                    # col = bpy.data.collections.get('Widgets')
                    # if not col:
                        # col = bpy.data.collections.new('Widgets')
                    obj = bpy.data.objects.get('WDGT_VIS')
                    if not obj:
                        obj = bpy.data.objects.new('WDGT_VIS',
                            bpy.data.meshes.new('WDGT_VIS'))
                    # if obj not in list(col.all_objects):
                        # col.objects.link(obj)
                    bone.custom_shape = obj
                elif wgt_op:
                    prefs = utils.prefs('zpy__mods').bone_widgets
                    keep = prefs.keep_settings
                    prefs.keep_settings = False
                    wgt_op(dict(active_object=bone.id_data, selected_pose_bones=[bone]),
                        mirror=False, relative_size=True, **args)
                    prefs.keep_settings = keep

            def lock(src, type, name=""):
                con = vis.constraints.new(type)
                con.target = src.id_data
                con.subtarget = src.name
                if name:
                    con.name = name
                con.show_expanded = False
                return con

            vis.bone.hide_select = True

            if self.type in ('ik_transforms',):  # IK + Copy Transforms:
                wgt(vis)
                if 'hand' in bone.name:
                    wgt(bone, widget='Blenrig - Hand')
                elif 'foot' in bone.name:
                    wgt(bone, widget='Blenrig - Foot')
                else:
                    wgt(bone, widget='Blenrig - IK Limb')
                    bone.custom_shape_transform = vis
                lock(src, 'COPY_TRANSFORMS', "Lock To Original")
            elif self.type in ('DAMPED_TRACK', 'LOCKED_TRACK', 'TRACK_TO', 'IK'):
                wgt(bone, widget='Sphere')
                wgt(vis)
                lock(bone, 'COPY_TRANSFORMS', "Lock To Copy")
                lock(src, 'COPY_ROTATION', "Lock To Original")
                bone.custom_shape_transform = vis
            elif self.type in ('STRETCH_TO',):
                wgt(bone, widget='Rigify - Shoulder')
                wgt(vis)
                lock(src, 'COPY_TRANSFORMS', "Lock To Original")
                bone.custom_shape_transform = vis
            elif self.type in ('FLOOR',):
                wgt(bone, widget='Plane', rotate=(radians(90), 0, 0))
                wgt(vis, widget='Blenrig - Box', rotate=(radians(90), 0, 0))
                lock(bone, 'COPY_TRANSFORMS', "Lock To Copy")
                lock(src, 'COPY_LOCATION', "Lock To Original")
            elif self.type in ('PIVOT',):
                wgt(bone, widget='ik_pole')
                wgt(vis)
                lock(src, 'COPY_TRANSFORMS', "Lock To Original")
            elif self.type in ('COPY_LOCATION', 'child_of_location'):
                # size = 1 + 0.2 * len([con for con in src.constraints
                                        #  if con.type in ('COPY_LOCATION', 'CHILD_OF')])
                size = 1.0
                wgt(bone, widget='Circle', global_size=size)
                wgt(vis)
                lock(src, 'COPY_TRANSFORMS', "Lock To Original")
                bone.custom_shape_transform = vis
            elif self.type in ('COPY_ROTATION', 'child_of_rotation'):
                # wgt(bone, widget='', slide=(0, 0.5, 0))
                # size = 0.25 + 0.2 * len([con for con in src.constraints
                                        #  if con.type in ('COPY_ROTATION', 'CHILD_OF')])
                size = 0.25
                wgt(bone, widget='Blenrig - Heel', slide=(-0.5, 0, 0), rotate=(0, 0, -radians(90)), global_size=size)
                wgt(vis)
                lock(src, 'COPY_TRANSFORMS', "Lock To Original")
                bone.custom_shape_transform = vis
            elif self.type in ('COPY_SCALE', 'child_of_scale'):
                # size = 1 + 0.2 * len([con for con in src.constraints
                                        #  if con.type in ('COPY_SCALE', 'CHILD_OF')])
                size = 1.0
                wgt(bone, widget='Finger', global_size=size)
                wgt(vis)
                lock(src, 'COPY_TRANSFORMS', "Lock To Original")
                bone.custom_shape_transform = vis
            elif self.type in ('COPY_TRANSFORMS', 'child_of_locrot'):
                # size = 1 + 0.2 * len([con for con in src.constraints
                                        #  if con.type in ('COPY_TRANSFORMS', 'CHILD_OF')])
                size = 1.0
                wgt(bone, widget='Rigify - Arm', global_size=size)
                wgt(vis)
                lock(src, 'COPY_TRANSFORMS', "Lock To Original")
                bone.custom_shape_transform = vis
            elif self.type in ('ARMATURE', 'CHILD_OF'):
                # size = 1 + 0.2 * len([con for con in src.constraints
                                        #  if con.type in ('ARMATURE', 'CHILD_OF')])
                size = 1.0
                wgt(bone, widget='Box', global_size=size)
                wgt(vis)
                # lock(src, 'COPY_TRANSFORMS', "Lock To Original")
                # bone.custom_shape_transform = vis
            # 'LIMIT_DISTANCE',
            # 'TRANSFORM',
            # 'SPLINE_IK',
            # 'ACTION',
            # 'pivot',  # Child of + Rotation
            # 'track_rotation',  # Damped Track + Rotation
            # 'stretch_rotation',  # Stretch To + Rotation
            elif self.type in ():
                wgt(bone, widget='Circle', slide=(0, 0.5, 0))
                wgt(vis)
                lock(src, 'COPY_TRANSFORMS', "Lock To Original")
                bone.custom_shape_transform = vis
            else:
                wgt(vis)
                lock(src, 'COPY_TRANSFORMS', "Lock To Original")
                bone.custom_shape_transform = vis

        def spline_ik(self, context, pairs):
            utils.update(context)
            ik_splines = dict()

            for (owner, target) in pairs:
                ik_splines[owner] = target

            spline_cons = set()

            for chain in Get.sorted_chains(ik_splines.keys()):
                if len(chain) == 1 and chain[0].parent:
                    continue
                    # owner = chain[0].parent_recursive[-1]
                    # if self.target == 'TO_ARMATURE':
                    #     target = Constraint.new_bone(self, owner)
                    # if self.target == 'TO_EMPTY':
                    #     target = Constraint.new_empty(self, owner)
                    # Constraint.add(self, owner, target, self.type)
                    # target.id_data.location = target.id_data.location

                    # chain.insert(0, owner)

                curve = New.curve(context, name=f"{Get.name(chain[-1])}_Spline_IK")
                spline = New.spline(curve)
                Set.visible(context, curve, False)

                for bone in chain:
                    target = ik_splines.get(bone, None)
                    if target is None:
                        continue

                    target.base_src.curve_target = curve.name

                    con = self.add_spline(context, bone, target, chain, ik_splines, curve, spline)
                    if con:
                        spline_cons.add(con)

                    if not bone.parent:
                        """Add movement to pelvis bone"""
                        con = self.new_constraint(bone, 'COPY_LOCATION')
                        self.set_constraint(context, con, bone, target)
                        spline_cons.add(con)

                # Keep base of spline attached to the rest of the rig
                bone = chain[0].parent
                target = ik_splines.get(chain[0], None)
                if (bone and target):
                    con = self.new_constraint(target, 'CHILD_OF')
                    self.set_constraint(context, con, target, bone)

            for con in spline_cons:
                con.influence = 1.0

            return

        def add_spline(self, context, bone, target, chain, ik_splines, curve, spline):

            def set_point(spline, index, co):
                point = spline.bezier_points[index]
                # point.handle_left_type = 'VECTOR'
                # point.handle_right_type = 'VECTOR'
                point.handle_left_type = 'AUTO'
                point.handle_right_type = 'AUTO'
                    # ('FREE', 'VECTOR', 'ALIGNED', 'AUTO')
                point.co = co

            # if Is.posebone(target):
                # from ._mods.boneWidget.functions.mainFunctions import createWidget
                # from ._mods.boneWidget.functions.jsonFunctions import readWidgets
                # createWidget(
                    # target,
                    # readWidgets()['Sphere'],
                    # False,  # self.relative_size,
                    # 0.2,  # self.global_size,
                    # (1.0, 1.0, 1.0),  # [*self.scale],  # [1, 1, 1],
                    # (0.0, 0.0, 0.0),  # self.slide,
                    # (0.0, 0.0, 0.0),  # self.rotate,
                # )
            if Is.empty(target):
                Set.empty_type(target, 'SPHERE')
                Set.empty_size(target, 0.02)

            # Store individual bone chains for each curve + chain
            ctxt = (repr(chain))
            spline_is_new = bool(ctxt not in ik_splines)

            if not spline_is_new:
                index = ik_splines[ctxt]
                spline.bezier_points.add(1)

            index = len(spline.bezier_points) - 1
            at_tail = bool(bone == chain[-1])
            ik_splines[ctxt] = index

            # Make curve match the shape of bones
            loc = Get.matrix(bone).to_translation()
            set_point(spline, index, loc)

            # Add hook modifier to curve
            mod = New.hook(curve, target, index)
            mod.name = f"Hook-{target.name}"

            if 1:  # not at_tail:  # or spline_is_new:
                """Add a point for every tail"""
                spline.bezier_points.add(1)

                # Make curve match the shape of bones
                loc = Get.matrix(bone).to_translation() + bone.tail - bone.head
                set_point(spline, index + 1, loc)

                # Add hook modifier to curve
                mod = New.hook(curve, target, index + 1)
                mod.name = f"Hook-{target.name}-Tail"
                utils.update(context)

            if not at_tail:
                return

            """Add Spline IK Constraint"""
            con = self.new_constraint(bone, 'SPLINE_IK')
            self.set_constraint(context, con, bone, curve)

            for b in [chain[-1]] + bone.parent_recursive:
                if b == chain[0]:
                    # con.chain_count += 1  # don't stop at first bone's tail
                    break
                con.chain_count += 1
            con.use_chain_offset = True  # Don't "move" bones to match spline

            con.y_scale_mode = 'BONE_ORIGINAL'  # 2.8
            # con.use_y_stretch = False  # 2.7

            con.xz_scale_mode = 'BONE_ORIGINAL'

            for chain_bone in chain:
                chain_target = ik_splines.get(chain_bone)

            # Keep the curve shape but make them flexible
            # update()
            for point in spline.bezier_points:
                point.handle_left_type = 'FREE'
                point.handle_right_type = 'FREE'
            con.influence = 0.0

            return con

        def init_nla(self, src, copy):
            blend = getattr(src.animation_data, 'action_blend_type', 'REPLACE')
            if blend != 'REPLACE':
                copy.animation_data_create().action_blend_type = blend

    class properties:
        """
        Constraint attributes for preferences and operator
        """
        type: EnumProperty(
            items=Constraint.constraint_type_items,
            name="Constraint Type",
            description="",
            default='COPY_TRANSFORMS',
            options={'SKIP_SAVE'},
        )
        target: EnumProperty(
            items=[
                ('', "To Selected", "", 'MOUSE_LMB', 0),
                ('TO_ACTIVE', "Active",
                    "Selected.constraints.target = Active",
                    'RESTRICT_SELECT_OFF', 2),
                ('TO_ARMATURE', "Armature",
                    "Selected.constraints.target = New_Rig_Bones",
                    'GROUP_BONE', 3),
                ('TO_OBJECT', "Object",
                    "Selected.constraints.target = New_Objects",
                    'OUTLINER_OB_EMPTY', 5),
                ('NONE', "No target",
                    "Add constraints without target",
                    'BLANK1', 1),
                ('', "From Selected", "", 'MOUSE_RMB', 0),
                ('FROM_ACTIVE', "Active",
                    "Active.constraints.target = Selected",
                    'OBJECT_DATAMODE', 6),
                ('FROM_ARMATURE', "Armature (dummies)",
                    "New_Rig_Bones.constraints.target = Selected",
                    'CONSTRAINT_BONE', 7),
                ('FROM_OBJECT', "Object (dummies)",
                    "New_Objects.constraints.target = Selected",
                    'OUTLINER_DATA_EMPTY', 9),
            ],
            name="Target Type",
            description="Default targetting mode for constraint creation",
            default='TO_ACTIVE',
            # options={'ENUM_FLAG'},
        )
        def new_arm_toggle(self, context):
            self.add_relation = self.new_armature
        new_armature: BoolProperty(
            name="Create New Armature Object(s)",
            description="Create new objects for bones, or use the original rig objects",
            default=True,
            update=new_arm_toggle,
        )

        armature_display_type: EnumProperty(
            items=[
                ('RANDOM', "Random",
                    "Randomly select the display type", 'FREEZE', 1),
                ('OCTAHEDRAL', "Octahedral",
                    "Display bones as octahedral shape (default)",
                    # 'PMARKER_SEL', 2),
                    'PMARKER_ACT', 2),
                ('STICK', "Stick",
                    "Display bones as simple 2D lines with dots",
                    'IPO_LINEAR', 3),
                ('BBONE', "B-Bone",
                    "Display bones as boxes, showing subdivision and B-Splines",
                    'MESH_CUBE', 5),
                ('ENVELOPE', "Envelope",
                    "Display bones as extruded spheres, showing deformation influence volume",
                    'PIVOT_MEDIAN', 6),
                ('WIRE', "Wire",
                    "Display bones as thin wires, showing subdivision and B-Splines",
                    'CURVE_DATA', 4),
            ],
            name="Display As",
            description="",
            default='BBONE',
            # options={'HIDDEN', 'SKIP_SAVE'}.
        )

        def set_head_tail(self, context):
            "Function to autoset At_Tail(Float), when setting Head_Tail(Bool)"

            self.head_tail = int(self.at_tail)
        at_tail: BoolProperty(
            name="Create At Tails",
            description="When creating new bones/objects, default them to "
                        "the originals' tail",
            default=False,
            options={'SKIP_SAVE'},
            update=set_head_tail,
        )
        head_tail: FloatProperty(
            name="Head/Tail",
            description="Target along length of bone: Head=0, Tail=1",
            default=0,
            soft_min=0,
            soft_max=1,
            step=3,  # (int) – Step of increment/decrement in UI, in [1, 100], defaults to 3 (* 0.01)
            precision=2,  # (int) – Maximum number of decimal digits to display, in [0, 6].
            options={'SKIP_SAVE'},
            subtype='FACTOR',  # (string) – Enumerator  in ['PIXEL', 'UNSIGNED', 'PERCENTAGE', 'FACTOR', 'ANGLE', 'TIME', 'DISTANCE', 'NONE'].
            # unit='NONE',  # (string) – Enumerator  in ['NONE', 'LENGTH', 'AREA', 'VOLUME', 'ROTATION', 'TIME', 'VELOCITY', 'ACCELERATION'].
        )

        copy_chain: BoolProperty(
            name="Copy Bone Parent Chain",
            description="Duplicate selected bones and "
                        "their parents and mimick their hierachy",
            default=False,
            options={'SKIP_SAVE'},
        )

        use_mesh: BoolProperty(
            name="Use Mesh Objects",
            description="When generating empty objects, create meshes instead",
            default=False,
            # options={'SKIP_SAVE'},
        )
        do_physics: BoolProperty(
            name="Add Phyics Objects",
            description="Create additional mesh objects with physics",
            default=False,
            options={'SKIP_SAVE'},
        )

        add_drivers: BoolProperty(
            name="Add Drivers",
            description="Insert drivers to the new constraints' influence",
            default=False,
            # options={'SKIP_SAVE'},
        )
        add_relation: BoolProperty(
            name="Add Relation",
            description="Insert custom properties to define relation "
                        "between the selected and new bones/objects",
            default=True,
            # options={'SKIP_SAVE'},
        )

        bake_animation: BoolProperty(
            name="Bake Transforms",
            # name="Bake Animation",
            description="Bake selected item's transforms to the new controls "
                        "before setting constraints",
            # description="Bake visual transforms to newly created bones/objects",
            default=False,
            # options={'SKIP_SAVE'},
        )
        bake_mode: EnumProperty(
            items=[
                # # ('frame', "Current Frame", "Store animation data from the current frame", 'PARTICLE_TIP', 0),
                ('range', "Frame Range", "Store animation data from the selected frame range", 'PARTICLE_PATH', 1),
                ('keyframes', "Available Keyframes", "Insert keys only on frames with existing keys (all transforms)", 'PARTICLE_POINT', 2),
            ],
            name="Cache Frames",
            description="Method to determine which frames to process for cache",
            default='range',
        )
        push_nla: BoolProperty(
            name="Push To NLA",
            description="Push baked actions to NLA strips after baking",
            default=False,
        )
        display_bake: BoolProperty(
            name="Display Bake Progress",
            description="Display baking process if baking animation at creation",
            default=True,
        )

        rotation: FloatVectorProperty(
            name="Rotation",
            description="Default rotation offset of new bones/objects",
            default=(0, 0, 0),
            soft_min=radians(-360),
            soft_max=radians(360),
            step=45,
            unit='ROTATION',
            size=3,
        )

        copy_chain: BoolProperty(
            name="Copy Bone Parent Chain",
            description="Duplicate selected bones and "
                        "their parents and mimick their hierachy",
            default=False,
            options={'SKIP_SAVE'},
        )

        def update_space(self, context):
            """Keep the spaces in sync, since I do this anyway"""
            self.owner_space = self.target_space
        target_space: EnumProperty(
            items=[
                ("WORLD", "World Space",
                    "The transformation of the target is evaluated relative to the world "
                    "coordinate system"),
                ("POSE", "Pose Space",
                    "The transformation of the target is only evaluated in the Pose Space, "
                    "the target armature object transformation is ignored"),
                ("LOCAL_WITH_PARENT",
                    "Local With Parent", "The transformation of the target bone is evaluated relative its local "
                    "coordinate system, with the parent transformation added"),
                ("LOCAL", "Local Space",
                    "The transformation of the target is evaluated relative to its local "
                    "coordinate system"),
            ],
            name="Target Space",
            description="",
            default=None,
            # options={},
            update=update_space,
        )
        owner_space: EnumProperty(
            items=[
                ("WORLD", "World Space",
                    "The constraint is applied relative to the world coordinate system"),
                ("POSE", "Pose Space",
                    "The constraint is applied in Pose Space, the object transformation is ignored"),
                ("LOCAL_WITH_PARENT", "Local With Parent",
                "The constraint is applied relative to the local coordinate system of the object, "
                    "with the parent transformation added"),
                ("LOCAL", "Local Space",
                    "The constraint is applied relative to the local coordinate system of the object"),
            ],
            name="Owner Space",
            description="",
            default=None,
            # options={},
        )
        use_local_location: BoolProperty(
            name="Local Location",
            description="When creating animation, use local space instead of world space",
            default=False,
        )

        track_axis: EnumProperty(
            items=[
                ("TRACK_X", "X", "", 'BLANKS1', 1),
                ("TRACK_Y", "Y", "", 'BLANKS1', 2),
                ("TRACK_Z", "Z", "", 'BLANKS1', 3),
                ("TRACK_NEGATIVE_X", "-X", "", 'BLANKS1', 4),
                ("TRACK_NEGATIVE_Y", "-Y", "", 'BLANKS1', 5),
                ("TRACK_NEGATIVE_Z", "-Z", "", 'BLANKS1', 6),
                ],
            name="Track Axis",
            description="",
            default='TRACK_Y',
            # options={},
        )
        track_offset: FloatProperty(
            name="Track Offset",
            description="Distance to offset damped trackers",
            default=0,
            min=0,
            soft_max=500,
            # step=100,
            precision=0,
            # options={'SKIP_SAVE'},
            subtype='PERCENTAGE',  # (string) – Enumerator  in ['PIXEL', 'UNSIGNED', 'PERCENTAGE', 'FACTOR', 'ANGLE', 'TIME', 'DISTANCE', 'NONE'].
            # unit='NONE',  # (string) – Enumerator  in ['NONE', 'LENGTH', 'AREA', 'VOLUME', 'ROTATION', 'TIME', 'VELOCITY', 'ACCELERATION'].
        )

        map_from: EnumProperty(
            items=[
                ('LOCATION', "Loc", ""),
                ('ROTATION', "Rot", ""),
                ('SCALE', "Scale", ""),
                ],
            name="Map From",
            description="Which parameter to set defaults for Transform constraint",
            default=None,
            options={'SKIP_SAVE'},
        )
        map_to: EnumProperty(
            items=[
                ('LOCATION', "Loc", ""),
                ('ROTATION', "Rot", ""),
                ('SCALE', "Scale", ""),
                ],
            name="Map To",
            description="Which parameter to set defaults for Transform constraint",
            default=None,
            options={'SKIP_SAVE'},
        )

        space_object: EnumProperty(
            items=[
                ("WORLD", "World Space",
                "The transformation of the target is evaluated"
                " relative to the world coordinate system"),
                ("LOCAL", "Local Space",
                "The transformation of the target is evaluated"
                " relative to its local coordinate system"),
                ],
            name="Space Object",
            description="",
            default=None,
            # options={},
        )

        show_expanded: BoolProperty(
            name="Expand Constraints",
            description="Leave constraint boxes open after creation",
            default=True,
        )

        create_widgets: BoolProperty(
            name="Create Widgets",
            description="Create Widget controllers for new bones",
            default=False,
        )

    # region: globals
    # Store the text label of constraint types and custom types
    constraint_labels_type = {x[1]: (x[0], x[2]) for x in (
        *Constraint.types.motion_tracking,
        *Constraint.types.transform,
        *Constraint.types.tracking,
        *Constraint.types.relationship,
    )}
    custom_labels_type = {x[1]: (x[0], x[2]) for x in (
        *Constraint.types.child_of,
        *Constraint.types.other,
    )}

    def _get_icons():
        """convert constraint_type_items into dict() of type and icon"""
        icons = dict()
        for entry, b, *c in Constraint.constraint_type_items:
            if len(c) == 1:
                icon = 'NONE'
            else:
                icon = c[1]
            icons[entry] = icon

        return icons
    icons = _get_icons()

    # endregion

    ###############################################################################
    # region: Functions

    def is_at_tail(src, other):
        for con in other.constraints:
            if not hasattr(con, 'target') or ((con.target != src) and (con.target != src.id_data or con.subtarget != src.name)):
                continue

            if (con.type == 'IK' and con.use_tail):
                return True
            elif con.type in ('DAMPED_TRACK', 'STRETCH_TO') and not con.head_tail:
                return True

        return False
    # def is_at_tail(src, other):
        # if other == get.original(src):
        #     return False
        # else:
        #     for con in other.constraints:
        #         if con.type in ('DAMPED_TRACK', 'STRETCH_TO', 'IK'):
        #             if con.target == src:
        #                 return True
        #             elif con.target == src.id_data and con.subtarget == src.name:
        #                 return True
        # if other == get.original(src):
        #     return src.constraints_relative.at_tail

        # connect = Constraint.get.references(src, other)
        # if connect:
        #     for rel in connect:
        #         if rel.at_tail:
        #             return True
        # else:
        #     for con in other.constraints:
        #         if con.type in ('DAMPED_TRACK', 'STRETCH_TO', 'IK'):
        #             if con.target == other.id_data and \
        #                     con.subtarget == other.name:
        #                 return True

    def get_copies(src):
        """
        return list of bones and objects created from the src
        """
        if not hasattr(src, 'base_src'): return

        copies = list()

        for item in src.base_src.copies:
            owner = (src.id_data if item.target_self else item.target)
            (target, subtarget) = Get.object_bone_from_string(owner, item.subtarget)

            copies.append(subtarget if item.subtarget else target)

        return copies

    def get_original(src):
        """
        return the original bone or object for the src
        """
        if not hasattr(src, 'base_src'): return

        base = src.base_src

        owner = (src.id_data if base.target_self else base.target)
        (target, subtarget) = Get.object_bone_from_string(owner, base.subtarget)

        return subtarget if base.subtarget else target

        # if hasattr(src.id_data, 'constraints_extra'):
            # con_list = src.id_data.constraints_extra
            # con_entry = con_list.get(src.name)
            # if con_entry is None:
                # return

            # co = con_entry.original
            # target, subtarget = Get.object_bone_from_string(
            # co.target, co.subtarget)

            # origin = subtarget if co.subtarget else target
        # else:
            # con_list = src.id_data.get('constraints_extra', dict())
            # con_entry = con_list.get(src.name, dict())
            # if con_entry is None:
                # return

            # co = con_entry['original']
            # target, subtarget = Get.object_bone_from_string(
            # co['target'], co['subtarget'])

            # origin = subtarget if co['subtarget'] else target

        # return origin

    # Was used in zpy_constraints\ui_properties_panel.py > draw_extra
    # def get_constraint_relations(context, src):
        # """
        # Add custom property to owner's object, to mark constraint information
        # """

        # # if not hasattr(src, 'base_src'): return

        # # return

        # # if hasattr(src.id_data, 'constraints_extra'):
            # # con_list = src.id_data.constraints_extra
            # # con_entry = con_list.get(src.name)
            # # if src.name in con_list:
            #     # con_entry = con_list[src.name]
            # # else:
            #     # return

            # # class extra:
            #     # constraints = con_entry.constraints
            #     # self = con_entry.self
            #     # copies = con_entry.copies
            #     # original = con_entry.original

            # # return extra

            # # # for con_name in con_constraints:
            #     # # item = con_constraints[con_name]
            #     # # owner = item['owner']
            #     # # target = item['target']
        # # else:
            # # con_list = src.id_data.get('constraints_extra', dict())
            # # con_entry = con_list.get(src.name)
            # # if con_entry is None:
            #     # return

            # # class extra:
            #     # constraints = con_entry.get('constraints', dict())
            #     # self = con_entry.get('self')
            #     # copies = con_entry.get('copies', dict())
            #     # original = con_entry.get('original')

            # # return extra

            # # # for con_name in con_constraints:
            #     # # item = con_constraints[con_name]
            #     # # owner = item['owner']
            #     # # target = item['target']

    # Was used in zpy_constraints\operator_remove_constraints
    # def remove_constraint_relation(context, name, *owners):
        # """
        # Delete custom properties of the constraint, in the owners' objects
        # """

        # # for src in owners:
            # # if (not src):
            #     # continue

            # # if not hasattr(src, 'base_src'): continue

            # # continue

            # # if hasattr(src.id_data, 'constraints_extra'):
                # # con_list = src.id_data.constraints_extra
                # # con_entry = con_list.get(src.name)
                # # if con_entry is None:
                #     # continue
                # # con_constraints = con_entry.constraints
                # # index = con_constraints.find(name)
                # # if index != -1:
                #     # con_constraints.remove(index)
            # # else:
                # # if (src.id_data.get('constraints_extra') is None):
                #     # continue
                # # con_list = Constraint.get_dict(src.id_data, 'constraints_extra')
                # # con_entry = Constraint.get_dict(con_list, src.name)
                # # con_constraints = Constraint.get_dict(con_entry, 'constraints')
                # # con_item = Constraint.get_dict(con_constraints, name)

                # # con_constraints.pop(name)
                # # if not con_constraints: con_entry.pop('constraints')
                # # if not con_entry: con_list.pop(src.name)
                # # if not con_list: src.id_data.pop('constraints_extra')

    # endregion
    ###############################################################################
