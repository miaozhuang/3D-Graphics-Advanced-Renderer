#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Pyglet GLSL Environmental Mapping on http://www.pythonstuff.org
# pythonian_at_gmx_dot_at (c) 2010
#
# based on the "graphics.py" batch/VBO demo by
# pyglet
# Copyright (c) 2006-2008 Alex Holkner
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------

'''Displays a rotating torus/sphere  using the pyglet.graphics API.
It demonstrates
    Environmental Cube Reflection Mapping on a bumpmapped surface
plus
    "scratches" by a texture map.
'''

html = '''
<font size=+3 color=#FF3030><br/>
<b>Environmental Bump Mapping in pyglet</b>
</font><br/>
<font size=+2 color=#00FF60>
R = Reset<br/>
Q, Esc = Quit<br/>
W, S, A, D = Up, Down, Left, Right<br/>
Space = Move/Stop<br/>
N, B = Normal Map Weight<br/>
Enter = Toggle Shader<br/>
T = Toggle Texture<br/>
Z = Toggle Bumpmap<br/>
F = Toggle Figure<br/>
H = This Help<br/>
</font>
'''

from math import pi, sin, cos, sqrt
from euclid import *

import pyglet
from pyglet.gl import *
from pyglet.window import key
from pyglet import image, resource

from shader import Shader

resource.path.append('textures')
resource.reindex()
texturecnt = 2

try:
    # Try and create a window with multisampling (antialiasing)
    config = Config(sample_buffers=1, samples=4,
                    depth_size=16, double_buffer=True,)
    window = pyglet.window.Window(resizable=True, config=config, vsync=False) # "vsync=False" to check the framerate
except pyglet.window.NoSuchConfigException:
    # Fall back to no multisampling for old hardware
    window = pyglet.window.Window(resizable=True)

label = pyglet.text.HTMLLabel(html, # location=location,
                              width=window.width//2,
                              multiline=True, anchor_x='center', anchor_y='center')

fps_display = pyglet.clock.ClockDisplay() # see programming guide pg 48

@window.event
def on_resize(width, height):
    if height==0: height=1
    # Keep text vertically centered in the window
    label.y = window.height // 2
    # Override the default on_resize handler to create a 3D projection
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., width / float(height), .1, 1000.)
    glMatrixMode(GL_MODELVIEW)
    return pyglet.event.EVENT_HANDLED

def update(dt):
    global autorotate
    global rot
    global dist

    if autorotate:
        rot += Vector3(0.1, 12, 5) * dt
        rot.x %= 360
        rot.y %= 360
        rot.z %= 360
pyglet.clock.schedule(update)

def dismiss_dialog(dt):
    global showdialog
    showdialog = False
pyglet.clock.schedule_once(dismiss_dialog, 10.0)

# Define a simple function to create ctypes arrays of floats:
def vec(*args):
    return (GLfloat * len(args))(*args)

@window.event
def on_draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, dist);
    glRotatef(rot.x, 0, 0, 1)
    glRotatef(rot.y, 0, 1, 0)
    glRotatef(rot.z, 1, 0, 0)

    glPolygonMode(GL_FRONT, GL_FILL)

    if shaderon:
        # bind our shader
        shader.bind()
        shader.uniformf('normalweight', normalweight )
        shader.uniformi('textureon', textureon )
        shader.uniformi('togglebump', togglebump )

        for i in range(texturecnt):
            glActiveTexture(GL_TEXTURE0+i)
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, texture[i].id)
            shader.uniformi('my_color_texture[' + str(i) + ']',i )

        glActiveTexture(GL_TEXTURE0+texturecnt)
        glEnable(GL_TEXTURE_CUBE_MAP)
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap)
        shader.uniformi('my_cube_texture', texturecnt)

        if togglefigure:
            batch1.draw()
        else:
            batch2.draw()

        for i in range(texturecnt):
            glActiveTexture(GL_TEXTURE0+i)
            glDisable(GL_TEXTURE_2D)
        glActiveTexture(GL_TEXTURE0+texturecnt)
        glDisable(GL_TEXTURE_CUBE_MAP)
        shader.unbind()
    else:
        if togglefigure:
            batch1.draw()
        else:
            batch2.draw()

    glActiveTexture(GL_TEXTURE0)
    glEnable(GL_TEXTURE_2D)
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    if showdialog:
        glLoadIdentity()
        glTranslatef(0, -200, -450)
        label.draw()

    glLoadIdentity()
    glTranslatef(250, -290, -500)
    fps_display.draw()

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glDisable(GL_TEXTURE_2D)

@window.event
def on_key_press(symbol, modifiers):
    global autorotate
    global rot
    global dist
    global textureon
    global togglebump
    global togglefigure
    global normalweight
    global light0pos
    global light1pos
    global shaderon
    global showdialog

    if symbol == key.R:
        print 'Reset'
        rot = Vector3(0, 0, 90)
    elif symbol == key.ESCAPE or symbol == key.Q:
        print 'Good Bye !'   # ESC would do it anyway, but not "Q"
        pyglet.app.exit()
        return pyglet.event.EVENT_HANDLED
    elif symbol == key.H:
        showdialog = not showdialog
    elif symbol == key.SPACE:
        print 'Toggle autorotate'
        autorotate = not autorotate
    elif symbol == key.ENTER:
        print 'Shader toggle'
        shaderon = not shaderon
    elif symbol == key.T:
        print 'Texture toggle'
        textureon = not textureon
    elif symbol == key.N:
        normalweight += 0.05
        print 'Normal Weight = ', normalweight
    elif symbol == key.B:
        normalweight -= 0.05
        print 'Normal Weight = ', normalweight
    elif symbol == key.Z:
        togglebump = not togglebump
        print 'Toggle Bump ', togglebump
    elif symbol == key.F:
        togglefigure = not togglefigure
        print 'Toggle Figure ', togglefigure
    elif symbol == key.PLUS:
        dist += 0.5
        print 'Distance now ', dist
    elif symbol == key.MINUS:
        dist -= 0.5
        print 'Distance now ', dist
    elif symbol == key.A:
        print 'Stop left'
        if autorotate:
            autorotate = False
        else:
            rot.y += -rotstep
            rot.y %= 360
    elif symbol == key.S:
        print 'Stop down'
        if autorotate:
            autorotate = False
        else:
            rot.z += rotstep
            rot.z %= 360
    elif symbol == key.W:
        print 'Stop up'
        if autorotate:
            autorotate = False
        else:
            rot.z += -rotstep
            rot.z %= 360
    elif symbol == key.D:
        print 'Stop right'
        if autorotate:
            autorotate = False
        else:
            rot.y += rotstep
            rot.y %= 360
    elif symbol == key.LEFT:
        print 'Light0 rotate left'
        tmp = light0pos[0]
        light0pos[0] = tmp * cos( lightstep ) - light0pos[2] * sin( lightstep )
        light0pos[2] = light0pos[2] * cos( lightstep ) + tmp * sin( lightstep )
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, vec(*light0pos))
    elif symbol == key.RIGHT:
        print 'Light0 rotate right'
        tmp = light0pos[0]
        light0pos[0] = tmp * cos( -lightstep ) - light0pos[2] * sin( -lightstep )
        light0pos[2] = light0pos[2] * cos( -lightstep ) + tmp * sin( -lightstep )
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, vec(*light0pos))
    elif symbol == key.UP:
        print 'Light0 up'
        tmp = light0pos[1]
        light0pos[1] = tmp * cos( -lightstep ) - light0pos[2] * sin( -lightstep )
        light0pos[2] = light0pos[2] * cos( -lightstep ) + tmp * sin( -lightstep )
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, vec(*light0pos))
    elif symbol == key.DOWN:
        print 'Light0 down'
        tmp = light0pos[1]
        light0pos[1] = tmp * cos( lightstep ) - light0pos[2] * sin( lightstep )
        light0pos[2] = light0pos[2] * cos( lightstep ) + tmp * sin( lightstep )
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, vec(*light0pos))
    else:
        print 'OTHER KEY'

def setup():
    # One-time GL setup
    global light0pos
    global light1pos
    global texture
    global cubemap

#   light0pos = [.5, .5, 1, 0]
    light0pos = [20.0,   20.0, 20.0, 1.0] # positional light !
    light1pos = [-20.0, -20.0, 20.0, 0.0] # infinitely away light !

    glClearColor(1, 1, 1, 1)
    glColor4f(1.0, 0.0, 0.0, 0.5 )
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glEnable(GL_TEXTURE_2D)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR ) #  GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR ) #  GL_NEAREST)

    texture = []
    for i in range (texturecnt):
        texturefile = 'Texturemap' + str(i) + '.jpg'
        print "Loading Texture", texturefile
        textureSurface = resource.texture(texturefile)
        texture.append( textureSurface )
        glBindTexture(GL_TEXTURE_2D, texture[i].id)
        print "Texture ", i, " bound to ", texture[i].id

    glEnable(GL_TEXTURE_CUBE_MAP)
    cubemap = GLuint()
    glGenTextures( 1, byref(cubemap))
    cubemap = cubemap.value
    print "CubeTexture is bound to", cubemap
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR ) #  GL_NEAREST)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR ) #  GL_NEAREST)

    cubename = ['left_wall_texture.jpg', 'right_wall_texture.jpg', 
                       'front_wall_texture.jpg', 'back_wall_texture3.jpg', 'pool_tex2.jpg', 'ceil_wall_tex.jpg']

    for i in range (6):
        cubefile = cubename[i]
        print "Loading Cube Texture", cubefile
        cube = resource.texture(cubefile) # instance of class AbstractImage
        data = cube.get_image_data().get_data('RGBA', cube.width * 4)

        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0,
                 GL_RGBA8, # texture.format, # 0,   # format
                 cube.width,
                 cube.height,
                 0,
                 GL_RGBA, GL_UNSIGNED_BYTE,
                 data)
    glDisable(GL_TEXTURE_CUBE_MAP)

    # Uncomment this line for a wireframe view
    #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    # Simple light setup.  On Windows GL_LIGHT0 is enabled by default,
    # but this is not the case on Linux or Mac, so remember to always
    # include it.
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)

    glLightfv(GL_LIGHT0, GL_POSITION, vec(*light0pos))
    glLightfv(GL_LIGHT0, GL_AMBIENT, vec(0.3, 0.3, 0.3, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(0.9, 0.9, 0.9, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, vec(1.0, 1.0, 1.0, 1.0))

    glLightfv(GL_LIGHT1, GL_POSITION, vec(*light1pos))
    glLightfv(GL_LIGHT1, GL_DIFFUSE, vec(.6, .6, .6, 1.0))
    glLightfv(GL_LIGHT1, GL_SPECULAR, vec(1.0, 1.0, 1.0, 1.0))

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(0.8, 0.5, 0.5, 1.0))
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(1, 1, 1, 1))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50)

#   glEnable(GL_COLOR_MATERIAL)

# create our Phong Shader by Jerome GUINOT aka 'JeGX' - jegx [at] ozone3d [dot] net
# see http://www.ozone3d.net/tutorials/glsl_lighting_phong.php

shader = Shader(['''
#version 110
varying vec3 lightVec, eyeVec;
varying mat3 TBNMatrix;

void main()
  {
  gl_Position = ftransform();
  gl_TexCoord[0]  = gl_TextureMatrix[0] * gl_MultiTexCoord0;

// Create the Texture Space Matrix
  vec3 normal  = normalize(gl_NormalMatrix * gl_Normal);
  vec3 tangent = normalize(gl_NormalMatrix * (gl_Color.rgb - 0.5));
  vec3 binormal = cross(normal, tangent);
  TBNMatrix = mat3(tangent, binormal, normal);

  vec3 position = vec3(gl_ModelViewMatrix * gl_Vertex);

// Compute the Eye Vector
  eyeVec  = (vec3(0.0) - position);
  eyeVec *= TBNMatrix;

// Compute the Light Vector
  lightVec  = gl_LightSource[0].position.xyz - position;
  lightVec *= TBNMatrix;
  }
'''], ['''
#version 110
varying vec3 lightVec, eyeVec;
uniform sampler2D my_color_texture['''+str(texturecnt)+'''];
uniform samplerCube my_cube_texture;

uniform int togglebump; // false/true
uniform int textureon; // false/true
uniform float normalweight;

varying mat3 TBNMatrix;

void main (void)
  {
  vec4 ambient  = vec4(0.0, 0.0, 0.0, 1.0); // all components neatly initialized
  vec4 diffuse  = vec4(0.0, 0.0, 0.0, 1.0);
  vec4 specular = vec4(0.0, 0.0, 0.0, 1.0);
  vec4 reflex   = vec4(0.0, 0.0, 0.0, 1.0);

// Compute parallax displaced texture coordinates
  vec3 eye = normalize(eyeVec);
  vec2 offsetdir = vec2( eye.x, eye.y );
  vec2 coords1 = gl_TexCoord[0].st;
  float dist = length(lightVec);
  vec3 light = normalize(lightVec);
  float attenuation = 1.0 / (gl_LightSource[0].constantAttenuation
                      + gl_LightSource[0].linearAttenuation * dist
                      + gl_LightSource[0].quadraticAttenuation * dist * dist);

// Query the Maps
  vec3 color = texture2D(my_color_texture[0], coords1).rgb;
  vec3 norm = vec3( 0.0, 0.0, 1.0 );
  if ( togglebump > 0 )
    {
    norm = normalize( texture2D(my_color_texture[1], coords1).rgb - 0.5);
    norm = vec3( norm.x * normalweight, norm.y * normalweight, norm.z );
    if ( length( norm.z ) < 0.001 )
      norm.z = 0.001;
    }
  vec3 refl = reflect(norm, eye);  // in tangent space !

  vec3 reflw = vec3( 1.0, -1.0, 1.0) * (TBNMatrix * refl);
  reflex = textureCube(my_cube_texture, reflw);

  if ( textureon > 0 )
    gl_FragColor = mix( reflex, vec4(color, 1.0), smoothstep( 0.7, 1.5, length(color)) );
  else
    gl_FragColor = reflex;
  }
'''])

class Torus(object):
    def __init__(self, radius, inner_radius, slices, inner_slices,
                 batch, group=None):
        # Create the vertex and normal arrays.
        vertices = []
        normals = []
        textureuvw = []
        tangents = []

        u_step = 2 * pi / (slices - 1)
        v_step = 2 * pi / (inner_slices - 1)
        u = 0.
        for i in range(slices):
            cos_u = cos(u)
            sin_u = sin(u)
            v = 0.
            for j in range(inner_slices):
                cos_v = cos(v)
                sin_v = sin(v)

                d = (radius + inner_radius * cos_v)
                x = d * cos_u
                y = d * sin_u
                z = inner_radius * sin_v

                nx = cos_u * cos_v
                ny = sin_u * cos_v
                nz = sin_v

                vertices.extend([x, y, z])
                normals.extend([nx, ny, nz])
                textureuvw.extend([u / (pi/1.5), v / (2.0* pi), 0.0])
                tangents.extend([ int(round(255 * (0.5 - 0.5 * sin_u))),
                                  int(round(255 * (0.5 + 0.5 * cos_u))),
                                  0 ])
                v += v_step
            u += u_step

        # Create a list of triangle indices.
        indices = []
        for i in range(slices - 1):
            for j in range(inner_slices - 1):
                p = i * inner_slices + j
                indices.extend([p, p + inner_slices, p + inner_slices + 1])
                indices.extend([p, p + inner_slices + 1, p + 1])

        self.vertex_list = batch.add_indexed(len(vertices)//3,
                                             GL_TRIANGLES,
                                             group,
                                             indices,
                                             ('v3f/static', vertices),
                                             ('n3f/static', normals),
                                             ('t3f/static', textureuvw),
                                             ('c3B/static', tangents))

    def delete(self):
        self.vertex_list.delete()

class Sphere(object):
    vv = []            # vertex vectors
    vcount = 0
    vertices = []
    normals = []
    textureuvw = []
    tangents = []
    indices  = []

    def myindex( self, list, value ):
        for idx, obj in enumerate(list):
            if abs(obj-value) < 0.0001:
              return idx
        raise ValueError # not found

    def splitTriangle(self, i1, i2, i3, new):
        '''
        Interpolates and Normalizes 3 Vectors p1, p2, p3.
        Result is an Array of 4 Triangles
        '''
        p12 = self.vv[i1] + self.vv[i2]
        p23 = self.vv[i2] + self.vv[i3]
        p31 = self.vv[i3] + self.vv[i1]
        p12.normalize()
        try:
            if new[0] == "X":
                ii0 = self.myindex(self.vv, p12)
            else:
                self.vv.append( p12 )
                ii0 = self.vcount
                self.vcount += 1
        except ValueError:
            print "This should not happen 1"
        p23.normalize()
        try:
            if new[1] == "X":
                ii1 = self.myindex(self.vv, p23)
            else:
                self.vv.append( p23 )
                ii1 = self.vcount
                self.vcount += 1
        except ValueError:
            print "This should not happen 2"
        p31.normalize()
        try:
            if new[2] == "X":
                ii2 = self.myindex(self.vv, p31)
            else:
                self.vv.append( p31 )
                ii2 = self.vcount
                self.vcount += 1
        except ValueError:
            print "This should not happen 3"
        rslt = []
        rslt.append([i1,  ii0, ii2])
        rslt.append([ii0, i2,  ii1])
        rslt.append([ii0, ii1, ii2])
        rslt.append([ii2, ii1,  i3])
        return rslt

    def recurseTriangle(self, i1, i2, i3, level, new):
        if level > 0:                     # split in 4 triangles
            p1, p2, p3, p4 = self.splitTriangle( i1, i2, i3, new )
            self.recurseTriangle( *p1, level=level-1, new=new[0]+"N"+new[2] )
            self.recurseTriangle( *p2, level=level-1, new=new[0]+new[1]+"N" )
            self.recurseTriangle( *p3, level=level-1, new="XNX" )
            self.recurseTriangle( *p4, level=level-1, new="X"+new[1]+new[2] )
        else:
           self.indices.extend( [i1, i2, i3] ) # just MAKE the triangle

    def flatten(self, x):
        """flatten(sequence) -> list

        Returns a single, flat list which contains all elements retrieved
        from the sequence and all recursively contained sub-sequences
        (iterables).

        Examples:
        >>> [1, 2, [3,4], (5,6)]
        [1, 2, [3, 4], (5, 6)]
        >>> flatten([[[1,2,3], (42,None)], [4,5], [6], 7, MyVector(8,9,10)])
        [1, 2, 3, 42, None, 4, 5, 6, 7, 8, 9, 10]"""

        result = []

        for el in x:
            #if isinstance(el, (list, tuple)):
            if hasattr(el, "__iter__") and not isinstance(el, basestring):
                result.extend(self.flatten(el))
            else:
                result.append(el)
        return result

    def __init__(self, radius, slices, batch, group=None):
        # Create the vertex array.
        self.vv.append( Vector3(1.0, 0.0, 0.0 ) ) # North
        self.vv.append( Vector3(-1.0, 0.0, 0.0) ) # South
        self.vv.append( Vector3(0.0, 1.0, 0.0 ) ) # A
        self.vv.append( Vector3(0.0, 0.0, 1.0 ) ) # B
        self.vv.append( Vector3(0.0, -1.0, 0.0) ) # C
        self.vv.append( Vector3(0.0, 0.0, -1.0) ) # D
        self.vcount = 6

        self.recurseTriangle( 0, 2, 3, slices, "NNN" ) # N=new edge, X=already done
        self.recurseTriangle( 0, 3, 4, slices, "XNN" )
        self.recurseTriangle( 0, 4, 5, slices, "XNN" )
        self.recurseTriangle( 0, 5, 2, slices, "XNX" )
        self.recurseTriangle( 1, 3, 2, slices, "NXN" )
        self.recurseTriangle( 1, 4, 3, slices, "NXX" )
        self.recurseTriangle( 1, 5, 4, slices, "NXX" )
        self.recurseTriangle( 1, 2, 5, slices, "XXX" )

        print "Sphere Level ", slices, " with ", self.vcount, " Vertices"

        for v in range(self.vcount):
            self.normals.extend(self.vv[v][:])
            # equal area projection, see http://www.uwgb.edu/dutchs/structge/sphproj.htm
            uv = Vector2(self.vv[v][2], self.vv[v][0])
            if abs(uv) > 1E-5:
                uv = uv.normalized() * abs( self.vv[v] + Vector3(0, -1, 0))
            self.textureuvw.extend([1.5-uv[1]/2.0, 1.5-uv[0]/2.0, 0.0])
            uvw = Vector3( self.vv[v][1], -self.vv[v][0], 0.0 ) # does not completely fit the ea-projection !
            if abs( uvw ) > 1E-5:
                uvw.normalize()
            self.tangents.extend([ int(round(255 * (0.5 - 0.5 * uvw[0]))),
                              int(round(255 * (0.5 - 0.5 * uvw[1]))),
                              int(round(255 * (0.5 - 0.5 * uvw[2]))) ])
        self.vv = [(x * radius) for x in self.vv]
        self.vertices = [x[:] for x in self.vv]
        self.vertices = self.flatten( self.vertices )

        self.vertex_list = batch.add_indexed(len(self.vertices)//3,
                                             GL_TRIANGLES,
                                             group,
                                             self.indices,
                                             ('v3f/static', self.vertices),
                                             ('n3f/static', self.normals),
                                             ('t3f/static', self.textureuvw),
                                             ('c3B/static', self.tangents))
    def delete(self):
        self.vertex_list.delete()

dist         = -3.5
rot          = Vector3(0, 0, 90)
autorotate   = True
rotstep      = 10
lightstep    = 10 * pi/180
shaderon     = True
textureon    = True
normalweight = 0.05
togglebump   = True
togglefigure = True
showdialog   = True

setup()

batch1  = pyglet.graphics.Batch()
torus = Torus(1, 0.3, 80, 25, batch=batch1)
batch2  = pyglet.graphics.Batch()
sphere = Sphere(1.5,3, batch=batch2)

pyglet.app.run()

#thats all