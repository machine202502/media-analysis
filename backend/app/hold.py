"""Stop signals for a video the viewer paused or deleted."""


class Held(Exception):
    """The viewer paused this video."""


class Gone(Exception):
    """The viewer deleted this video."""
