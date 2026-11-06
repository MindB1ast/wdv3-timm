"""Defaults for new rule widgets restored from older serialized workflows."""


def child_route_default(value):
    return 'separate' if value is None or isinstance(value, str) and not value.strip() else value


def tag_enabled_default(value):
    return True if value is None or isinstance(value, str) and not value.strip() else value
