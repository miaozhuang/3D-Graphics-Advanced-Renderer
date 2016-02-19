

#from OpenGLContext import testingcontext
#BaseContext = testingcontext.getInteractive()
#from OpenGLContext.arrays import *
#from OpenGLContext.events.timer import Timer
#from OpenGLContext.scenegraph.quadrics import Sphere
from OpenGL.arrays import vbo
from OpenGL import *
from OpenGL._null import NULL
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.constants import *
from OpenGL.constants import GLfloat_3, GLfloat_4
from numpy import array
try:
    from PIL.Image import open as pil_open
except ImportError, err:
    from Image import open as pil_open
