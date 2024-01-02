try:
    import matplotlib
    matplotlib.use('tkagg')  # prevent issues with cv2 in combination with the Qt platform plugin "xcb"
except ImportError:
    import warnings
    warnings.warn("Could not set matplotlib backend to 'tkagg'. This might cause issues with cv2 in combination with the Qt platform plugin 'xcb'.")

__version__ = '0.2.0'
