import os
from matplotlib.style.core import USER_LIBRARY_PATHS, reload_library

# get path to styles
styles_dir = os.path.join(os.path.dirname(__file__), 'styles')

# add styles to mpl
USER_LIBRARY_PATHS.append(styles_dir)
reload_library()
