#! /usr/bin/env python
import OpenGL 
from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
from OpenGL.raw.GL.ARB.framebuffer_object import GL_FRAMEBUFFER
from numpy import array

from core.materal_parser import MaterialParser
from core.render import Render, Camera


#from core.gui import GUI
OpenGL.ERROR_ON_COPY = True 

# PyOpenGL 3.0.1 introduces this convenience module...


# width, height = 640, 480  # initial window size
width, height = 720, 540  # initial window size
# width, height = 800, 600
degree = 0
vertex_data = None

# A general OpenGL initialization function.  Sets all of the initial parameters. 
def InitGL(Width, Height):  # We call this right after our OpenGL window is created.
    global vertex_data
    # glClearColor(0.0, 0.0, 0.0, 0.0)  # This Will Clear The Background Color To Black
    glClearColor(1.0, 1.0, 1.0, 1.0)  # This Will Clear The Background Color To Black
    glClearDepth(1.0)  # Enables Clearing Of The Depth Buffer
    glDepthFunc(GL_LEQUAL)  # The Type Of Depth Test To Do
    glEnable(GL_DEPTH_TEST)  # Enables Depth Testing
    glShadeModel(GL_SMOOTH)  # Enables Smooth Color Shading
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()  # Reset The Projection Matrix
    # Calculate The Aspect Ratio Of The Window
    gluPerspective(45.0, float(Width) / float(Height), 0.1, 50.0)

    glMatrixMode(GL_MODELVIEW)
 
    # get shader program
    Render.trace_program = Render.get_shader_program(
        "shaders/main.vs.glsl", "shaders/main.fs.glsl")
    Render.draw_program = Render.get_shader_program(
        "shaders/display.vs.glsl", "shaders/display.fs.glsl")
    # prepare data
    vertex_data = vbo.VBO(
        array([
               [1.0, 1.0],
               [-1.0, 1.0],
               [1.0, -1.0],
               [-1.0, -1.0]
        ], 'f')
    )
    # get attr_locs of vertex attributes
    # location
    loc_map = {"camera.fov_factor" : "cam_fov_loc",
                   "camera.res" : "cam_res_loc",
                   "camera.pos" : "cam_pos_loc",
                   "trans" : "cam_trans_loc",
                   "sample_count" : "sample_count_loc",
                   "prev_tex" : "prev_tex_loc",
                   "mtl_tex" : "mtl_tex_loc",
                   "mtl_num" : "mtl_num_loc",
                   "front_wall_tex" : "front_wall_tex_loc",
                   "back_wall_tex" : "back_wall_tex_loc",
                   "left_wall_tex" : "left_wall_tex_loc",
                   "right_wall_tex" : "right_wall_tex_loc",
				   "ceil_wall_tex" : "ceil_wall_tex_loc",
                   "pool_tex" : "pool_tex_loc",
                   "water_norm_0" : "water_norm0_loc",
                   "water_norm_1" : "water_norm1_loc",
                   "glob_time" : "glob_time_loc",
                   "pause" : "stop_motion_loc",
                   "refresh" : "refresh_loc"
                   } 
    Render.init_uniforms(loc_map)
    
    
    # set camera
    Camera.res = [width, height]
    Camera.update_rtrans()
    
    # load material
    material_file = "res/material_data"
    Render.mtl_num, Render.mtl_tex = Render.make_texture_float(loadMaterial(material_file))
    print("material data....")
    print(Render.mtl_num, Render.mtl_tex)
   
    # create empty textures for rendering
    Render.texs.append(Render.make_texture())
    Render.texs.append(Render.make_texture())
    
    # create image textures
    for image_file in ['left_wall_texture.jpg', 'right_wall_texture.jpg', 
                       'front_wall_texture.jpg', 'back_wall_texture3.jpg', 'pool_tex2.jpg', 'ceil_wall_tex.jpg']:
        Render.image_texs.append(Render.get_texture("res/%s" % (image_file)))
            
    # create water norms
    for norm_file in ['cloud_norm1.jpg', 'cloud_norm2.jpg']:
        Render.water_norms.append(Render.get_texture("res/%s" % (norm_file)))
        
    # Framebuffer
    Render.fbo = glGenFramebuffers(1)
    if not glBindFramebuffer:
        print ("Framebuffer not supported")
                    
def loadMaterial(material_file):
    # load material
    material_data = []
    # build material data as texture
    raw_data = MaterialParser(material_file).parse();
    keys = ['Ka', 'Kd', 'Ks', 'illum', 'Ns', 'alpha']
    for k, v in raw_data:
        v['illum'][0] /= 100.0
        v['Ns'][0] /= 100.0
        for key in keys:
            for item in v[key]:
                material_data.append(item)
    return material_data

# The function called when our window is resized (which shouldn't happen if you enable fullscreen, below)
def ReSizeGLScene(Width, Height):
    global width, height
    glutReshapeWindow(width, height);
    return
    if Height == 0:  # Prevent A Divide By Zero If The Window Is Too Small 
        Height = 1

    width = Width
    height = Height
    # set refresh
    Render.refresh = True
    # set camera
    Camera.res = [width, height]
    Camera.update_rtrans()
    glViewport(0, 0, Width, Height)  # Reset The Current Viewport And Perspective Transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(Width) / float(Height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

# The main_old drawing function. 
def DrawGLScene():
    global degree
    # global vertex_data
    # Clear The Screen And The Depth Buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()  # Reset The View 

    render_to_texture()

    display()

    #  since this is double buffered, swap the buffers to display what just got drawn. 
    glutSwapBuffers()

def render_to_texture():
    glUseProgram(Render.trace_program)
    glUniform1f(getattr(Render.trace_program, 'mtl_num_loc'), Render.mtl_num * 1.0)
    # textures
    tex_data = [
                (Render.image_texs[0], 'left_wall_tex_loc', GL_TEXTURE2, 2),
                (Render.image_texs[1], 'right_wall_tex_loc', GL_TEXTURE3, 3),
                (Render.image_texs[2], 'front_wall_tex_loc', GL_TEXTURE4, 4),
                (Render.image_texs[3], 'back_wall_tex_loc', GL_TEXTURE5, 5),
                (Render.image_texs[4], 'pool_tex_loc', GL_TEXTURE6, 6),
				(Render.image_texs[5], 'ceil_wall_tex_loc', GL_TEXTURE9, 9),
                (Render.water_norms[0], 'water_norm0_loc', GL_TEXTURE7, 7),
                (Render.water_norms[1], 'water_norm1_loc', GL_TEXTURE8, 8),
                (Render.mtl_tex, 'mtl_tex_loc', GL_TEXTURE1, 1),
                (Render.texs[0], 'prev_tex_loc', GL_TEXTURE0, 0),
                ]
    for t in tex_data:
        glActiveTexture(t[2])
        glBindTexture(GL_TEXTURE_2D, t[0])
        glUniform1i(getattr(Render.trace_program, t[1]), t[3])
         
    Camera.update_rtrans()
    Render.sample_count += 1

    Render.update_program()
    glBindFramebuffer(GL_FRAMEBUFFER, Render.fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, Render.texs[1], 0)
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        print("Framebuffer failed")
    glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])
    #glDrawBuffer(GL_NONE)
    glBindFramebuffer(GL_FRAMEBUFFER, Render.fbo)
    #glBindFramebuffer(GL_FRAMEBUFFER, 0)
    # render to texture
    pos_loc = glGetAttribLocation(Render.trace_program, 'pos')
    try:
        vertex_data.bind()
        glEnableVertexAttribArray(pos_loc)
        stride = vertex_data.data[0].nbytes
        glVertexAttribPointer(pos_loc, 2, GL_FLOAT, False, stride, vertex_data)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
    finally:
        glDisableVertexAttribArray(pos_loc)
        vertex_data.unbind()
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    # Swap the texture buffer
    Render.texs.reverse()
    
def display():
    glUseProgram(Render.draw_program)
    #glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, Render.texs[1])
    pos_loc = glGetAttribLocation(Render.draw_program, 'pos')
    try:
        vertex_data.bind()
        glEnableVertexAttribArray(pos_loc)
        stride = vertex_data.data[0].nbytes
        glVertexAttribPointer(pos_loc, 2, GL_FLOAT, False, stride, vertex_data)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
    finally:
        glDisableVertexAttribArray(pos_loc)
        vertex_data.unbind()
    
def show_case(no):
    positions = [
                 [7.0, 2.0, 3.0],
                 [-5.,  -3.2, -4.2],
                 [0.6,  -4.4, -14.2],
                 [ 5.,   4.,  -4.6],
                 [5.,   4.,   .2],
                 [4.2,   1.6,  19.8]
                 ]
    rotates = [
               [-7.0, 25.0, 0.0], 
               [-31.7  -6.8, 0.0], 
               [-29., -20, 0], 
               [-26.5, 0, 0], 
               [-64.1, 3.14, 0], 
               [-0.88, 157.0, 0]
               ]
    Camera.pos = array(positions[no])
    Camera.rotate = array(rotates[no])
    
def print_camera():
    print(Camera.pos)
    print(Camera.rotate)
# The function called whenever a key is pressed. Note the use of Python tuples to pass in: (key, x, y)  
def keyPressed(*args):
    # If escape is pressed, kill everything.
    if args[0] == '\x1b':
        sys.exit()
    speed = 0.4
    key = args[0].upper()
    if key == 'W':
        Camera.pos[2] += speed
	if Camera.pos[2] > 15.0:
		Camera.pos[2] = 15.0
    elif key == 'S':
        Camera.pos[2] -= speed
	if Camera.pos[2] < -15.0:
		Camera.pos[2] = -15.0
    elif key == 'A': 
        Camera.pos[0] -= speed
	if Camera.pos[0] < -7.0:
		Camera.pos[0] = -7.0
    elif key == 'D':
        Camera.pos[0] += speed
	if Camera.pos[0] > 9.0:
		Camera.pos[0] = 9.0
    elif key == 'Q':
        Camera.pos[1] += speed
	if Camera.pos[1] > 9.0:
		Camera.pos[1] = 9.0
    elif key == 'E':
        Camera.pos[1] -= speed
	if Camera.pos[1] < -9.0:
		Camera.pos[1] = -9.0
    elif key == ' ':
        Render.pause = not Render.pause
    elif key == '0':
        show_case(0)
    elif key == '1':
        show_case(1)
    elif key == '2':
        show_case(2)
    elif key == '3':
        show_case(3)
    elif key == '4':
        show_case(4)
    elif key == '5':
        show_case(5)
    elif key == 'P':
        print_camera()
	if Camera.pos[2] < 1:
	    Camera.pos[2] = 1

    Render.sample_count = 0

def mouseAction(*args):
    global drag, oldx, oldy
    if args[0] == GLUT_LEFT_BUTTON:
        if args[1] == GLUT_UP:
            drag = False 
        else:
            drag = True 
            oldx = args[2]
            oldy = args[3]
            
def mouseMove(*args):
    global oldx, oldy
    drag_speed = 200.0
    if drag:
        Camera.rotate[1] += (args[0] - oldx) * drag_speed / height
        Camera.rotate[0] += (args[1] - oldy) * drag_speed / width
        if Camera.rotate[0] > 89:Camera.rotate[0] = 89
        if Camera.rotate[1] > 180:Camera.rotate[1] -= 360
        if Camera.rotate[0] < -89:Camera.rotate[0] = -89
        if Camera.rotate[1] < -180:Camera.rotate[1] += 360
        oldx = args[0]
        oldy = args[1]
        Render.sample_count = 0
#         print Camera.rotate


def main():
    global window
    # For now we just pass glutInit one empty argument. I wasn't sure what should or could be passed in (tuple, list, ...)
    # Once I find out the right stuff based on reading the PyOpenGL source, I'll address this.
    glutInit(sys.argv)
    # Select type of Display mode:   
    #  Double buffer 
    #  RGBA color
    # Alpha components supported 
    # Depth buffer
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    
    # get a 640 x 480 window 
    glutInitWindowSize(width, height)
    
    # the window starts at the upper left corner of the screen 
    glutInitWindowPosition(0, 0)
    
    # Okay, like the C version we retain the window id to use when closing, but for those of you new
    # to Python (like myself), remember this assignment would make the variable local and not global
    # if it weren't for the global declaration at the start of main_old.
    window = glutCreateWindow("Advanced Renderer")

    # Register the drawing function with glut, BUT in Python land, at least using PyOpenGL, we need to
    # set the function pointer and invoke a function to actually register the callback, otherwise it
    # would be very much like the C version of the code.    
    glutDisplayFunc(DrawGLScene)
    
    # Uncomment this line to get full screen.
#     glutFullScreen()

    # When we are doing nothing, redraw the scene.
    glutIdleFunc(DrawGLScene)
    
    # Register the function called when our window is resized.
    glutReshapeFunc(ReSizeGLScene)
    
    # Register the function called when the keyboard is pressed.  
    glutKeyboardFunc(keyPressed)
    
    glutMouseFunc(mouseAction)
    glutMotionFunc(mouseMove)

    # Initialize our window. 
    InitGL(width, height)

    # Start Event Processing Engine    
    glutMainLoop()

# Print message to console, and kick off the main_old to get it rolling.

if __name__ == "__main__":
    print('''
    Shortcuts:
    [Esc]: Quit the program
    [W]: Forward
    [S]: Backward
    [A]: Left
    [D]: Right
    [Q]: Up
    [E]: Down
    ''')
    main()
