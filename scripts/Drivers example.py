# Drivers example

if __name__ == '__main__':
    import bpy

    attributes = {}

    def drv_delay(frame, attr, delay, value):
        """Used to delay the value by a specified number of frames."""

        # Determine whether we already have a value for this attribute
        if attr in attributes:
            attribute = attributes[attr]
        else:
            # Not found - create a new record for it and store it
            attribute = {'frame': frame, 'dataqueue': []}
            attributes[attr] = attribute

        if frame <= 1:
            del(attribute['dataqueue'][:])

        # Only store the value on change of frame (to guard against being called multiple times per frame)
        if frame != attribute['frame']:

            # Store this value in the queue
            attribute['dataqueue'].append(value)

            # Discard any elements that are more than the delay period
            while len(attribute['dataqueue']) > (delay + 1):
                del(attribute['dataqueue'][0])  # remove one from head of the list

        # Store frame
        attributes[attr]['frame'] = frame

        #return the value at the head of the list
        if len(attribute['dataqueue']) == 0:
            return value
        else:
            return attribute['dataqueue'][0]

    if 'drv_delay' in bpy.app.driver_namespace:
        del bpy.app.driver_namespace['drv_delay']
    bpy.app.driver_namespace['drv_delay'] = drv_delay
