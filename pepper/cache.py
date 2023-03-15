from functools import wraps


# From Tidy3D repo
# https://github.com/flexcompute/tidy3d/blob/5d16f2ec9c7f08533d6c93d2cb0c5c64dcfb956a/tidy3d/components/base.py
def cache(prop):
    """Decorates a property to cache the first computed value and return it on subsequent calls."""

    # note, we could also just use `prop` as dict key, but hashing property might be slow
    prop_name = prop.__name__

    @wraps(prop)
    def cached_property_getter(self):
        """The new property method to be returned by decorator."""

        stored_value = self._cached_properties.get(prop_name)  # pylint:disable=protected-access

        if stored_value is not None:
            return stored_value

        computed_value = prop(self)
        self._cached_properties[prop_name] = computed_value  # pylint:disable=protected-access
        return computed_value

    return cached_property_getter


def cached_property(cached_property_getter):
    """Shortcut for property(cache()) of a getter."""

    return property(cache(cached_property_getter))
