try:
    import rclpy  # noqa: F401
except ImportError:
    raise RuntimeError("Could not find 'rclpy'. Did you forget to source ROS?")
