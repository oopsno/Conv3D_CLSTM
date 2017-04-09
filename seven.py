"""
Seven means `six` and a little bit more
"""

import six

# String IO
if six.PY2:
    from cStringIO import StringIO
else:
    from io import StringIO

# xrange
if not six.PY2:
    xrange = range
else:
    xrange = xrange

