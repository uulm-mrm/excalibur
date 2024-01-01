import matplotlib.colors as mcolors


def get_color_array(name, alpha=False):
    named_colors = mcolors.get_named_colors_mapping()
    if name in named_colors:
        color = mcolors.to_rgba_array(named_colors[name]).squeeze()
        if not alpha:
            color = color[:3]
        return color
    else:
        raise RuntimeError(f"Named color '{name}' not found")


def get_color_list(name, alpha=False):
    return get_color_array(name, alpha=alpha).tolist()


def get_color_tuple(name, alpha=False):
    return tuple(get_color_list(name, alpha=alpha))
