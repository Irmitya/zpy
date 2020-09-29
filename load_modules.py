import bpy

# __all__ = (
    # # "init",
    # # "register",
    # # "unregister",
    # "load_modules",
# )


class load_modules():
    def __init__(self, file, package):
        self.__file__ = file
        self.__package__ = package
        self.modules = None
        self.ordered_classes = None
        self.prefs_props = list()
        self.keymaps = dict()

    def init(self):
        from pathlib import Path

        self.modules = self.get_all_submodules(Path(self.__file__).parent)
        self.ordered_classes = self.get_ordered_classes_to_register(self.modules)

    def register(self):
        self.delete_zpy()
        self.init()

        for module in self.modules:
            if hasattr(module, "classes"):
                for cls in list(module.classes):
                    if getattr(cls, "is_registered", False):
                        bpy.utils.unregister_class(cls)
                    try:
                        self.register_class(cls)
                    except:
                        pass

        failed_classes = list()
        for cls in self.ordered_classes:
            if not getattr(cls, "is_registered", False):
                try:
                    self.register_class(cls)

                    # Send the keymaps directly to classes
                    if (bpy.types.AddonPreferences in cls.mro()):
                        cls.keymaps = self.keymaps
                except:
                    failed_classes.append(cls)

        for module in self.modules.copy():
            if module.__name__ == __name__:
                continue
            if hasattr(module, "register"):
                try:
                    module.register()
                except Exception as error:
                    print(f"Can't register module:\t{module.__name__}\n{error}")
                    self.modules.remove(module)

                    if hasattr(module, "classes"):
                        for cls in list(module.classes):
                            if getattr(cls, "is_registered", False):
                                bpy.utils.unregister_class(cls)
                    continue
            if hasattr(module, 'km'):
                for keymap, kmis in module.km.addon_keymaps.items():
                    if keymap not in self.keymaps:
                        self.keymaps[keymap] = list()
                    for kmi in kmis:
                        if kmi not in self.keymaps[keymap]:
                            self.keymaps[keymap].append(kmi)

        for cls in failed_classes:
            if not getattr(cls, "is_registered", False):
                try:
                    self.register_class(cls)
                except Exception as error:
                    self.ordered_classes.remove(cls)
                    print("Error: Can't register class: ", cls)
                    print("\t", error)

    def register_class(self, cls):
        # bpy_props = tuple(eval(f'bpy.props.{x}Property') for x in (
        #     'Pointer', 'Collection',
        #     'Bool', 'BoolVector', 'Enum', 'String',
        #     'Float', 'FloatVector', 'Int', 'IntVector',
        # ))
        if bpy.app.version < (2, 80, 0):
            # De-annotate properties for 2.7 backport
            for attribute in getattr(cls, '__annotations__', dict()).copy():
                prop, kwargs = cls.__annotations__[attribute]
                if not hasattr(cls, attribute):
                    setattr(cls, attribute, prop(**kwargs))
                    if attribute not in cls.order:
                        cls.order.append(attribute)
        # else:
            # # Annotate properties for 2.8
            # for attr in dir(cls):
            #     prop = getattr(cls, attr)

            #     if not prop:
            #         continue
            #     elif isinstance(prop, tuple) and prop[0] in bpy_props:
            #         pass
            #     elif prop in bpy_props:
            #         pass
            #     else:
            #         continue

            #     if not hasattr(cls, '__annotations__'):
            #         cls.__annotations__ = dict()

            #     if not cls.__annotations__.get(attr):
            #         # If the property is already annotated, overwriting it may not be desired
            #         cls.__annotations__[attr] = prop
            #         delattr(cls, attr)

        bpy.utils.register_class(cls)

    def unregister(self):
        from sys import modules as sys_modules

        for cls in reversed(self.prefs_props):
            if getattr(cls, 'is_registered', False):
                bpy.utils.unregister_class(cls)
            self.prefs_props.remove(cls)

        for cls in reversed(self.ordered_classes):
            if getattr(cls, 'is_registered', False):
                bpy.utils.unregister_class(cls)

        for module in self.modules:
            if module.__name__ == __name__:
                continue
            if hasattr(module, "unregister"):
                try:
                    module.unregister()
                except Exception as error:
                    print(f"Can't unregister module:\t{module.__name__}\n{error}")

        for module in self.modules:
            if module.__name__ == __name__:
                continue
            if module.__name__ in sys_modules:
                del (sys_modules[module.__name__])

        # Remove the remaining entries for the folder, zpy, and zpy.functions
        for module_name in reversed(list(sys_modules.keys())):
            # if module_name == __name__:  # This should exist anyway
            #     continue
            if module_name.startswith(self.__package__ + '.') or module_name == self.__package__:
                del sys_modules[module_name]

        self.keymaps.clear()

    # Import modules
    #################################################

    def get_all_submodules(self, directory):
        return list(self.iter_submodules(directory, directory.name))

    def iter_submodules(self, path, package_name):
        import importlib
        for name in sorted(self.iter_submodule_names(path)):
            # try:
            yield importlib.import_module("." + name, package_name)
            # except Exception as error:
                # print(error, name, package_name, path)

    def iter_submodule_names(self, path, root=""):
        import pkgutil
        for _, module_name, is_package in pkgutil.iter_modules([str(path)]):
            if is_package:
                sub_path = path / module_name
                sub_root = root + module_name + "."
                yield from self.iter_submodule_names(sub_path, sub_root)
            else:
                yield root + module_name

    def delete_zpy(self):
        "Delete zpy so that it can reload"
        from sys import modules as sys_modules

        if self.modules is not None:
            for module in self.modules:
                if module.__name__ == __name__:
                    continue
                if module.__name__ in sys_modules:
                    del (sys_modules[module.__name__])

        root = None
        # from pathlib import Path
        # root_self = str(Path(self.__file__).parent) + '\\'
        for (name, module) in sys_modules.copy().items():
            if getattr(module, '__file__', None) is None:
                continue

            if name == 'zpy' and hasattr(module, '__path__'):
                root = module.__path__[0] + '\\'

            if (root is not None and module.__file__.startswith(root)):
                del sys_modules[name]

    # Find classes to register
    #################################################

    def get_ordered_classes_to_register(self, modules):
        return self.toposort(self.get_register_deps_dict(modules))

    def get_register_deps_dict(self, modules):
        deps_dict = {}
        classes_to_register = set(self.iter_classes_to_register(modules))
        for cls in classes_to_register:
            deps_dict[cls] = set(self.iter_own_register_deps(cls, classes_to_register))
        return deps_dict

    def iter_own_register_deps(self, cls, own_classes):
        yield from (dep for dep in self.iter_register_deps(cls) if dep in own_classes)

    def iter_register_deps(self, cls):
        import typing
        for value in typing.get_type_hints(cls, {}, {}).values():
            dependency = self.get_dependency_from_annotation(value)
            if dependency is not None:
                yield dependency

    def get_dependency_from_annotation(self, value):
        if isinstance(value, tuple) and len(value) == 2:
            if value[0] in (bpy.props.PointerProperty, bpy.props.CollectionProperty):
                return value[1]["type"]
        return None

    def iter_classes_to_register(self, modules):
        base_types = self.get_register_base_types()
        for cls in self.get_classes_in_modules(modules):
            # if any(base in base_types for base in cls.__bases__):
            if any(base in base_types for base in cls.mro()):
                if not getattr(cls, "is_registered", False):
                    # if bpy.types.AddonPreferences in cls.__bases__:
                    #     if not hasattr(cls, '__annotations__'):
                    #         cls.__annotations__ = dict()
                    #     self.reg_addon_groups(cls)
                    yield cls

    def reg_addon_groups(self, cls):
        "Find PropertyGroup classes and register them as Pointers"
        from inspect import isclass

        for i in vars(cls):
            prop = getattr(cls, i, None)
            if not isclass(prop):
                continue

            if issubclass(prop, bpy.types.PropertyGroup):
                if not getattr(prop, "is_registered", False):
                    register_class(prop)
                    self.prefs_props.append(prop)
                self.reg_addon_groups(prop)
                group = bpy.props.PointerProperty(type=prop)

                # setattr(cls, i, group)
                if hasattr(cls, '__annotations__'):
                    cls.__annotations__[i] = group
                else:
                    setattr(cls, i, group)

    def get_classes_in_modules(self, modules):
        classes = set()
        for module in modules:
            for cls in self.iter_classes_in_module(module):
                classes.add(cls)
        return classes

    def iter_classes_in_module(self, module):
        from inspect import isclass

        for value in module.__dict__.values():
            if isclass(value):
                yield value

    def get_register_base_types(self):
        return set(getattr(bpy.types, name) for name in [
            "Panel", "Operator", "PropertyGroup",
            "AddonPreferences", "Header", "Menu",
            "Node", "NodeSocket", "NodeTree",
            "UIList", "RenderEngine",
            'KeyingSetInfo',
        ])

    # Find order to register to solve dependencies
    #################################################

    def toposort(self, deps_dict):
        sorted_list = []
        sorted_values = set()
        while len(deps_dict) > 0:
            unsorted = []
            for value, deps in deps_dict.items():
                if len(deps) == 0:
                    sorted_list.append(value)
                    sorted_values.add(value)
                else:
                    unsorted.append(value)
            deps_dict = {value: deps_dict[value] - sorted_values for value in unsorted}
        return sorted_list
