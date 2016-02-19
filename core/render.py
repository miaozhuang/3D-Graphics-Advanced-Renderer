
import math

from core.opengl_import import *
from core.util import Util
import numpy as np



class Camera(object):
    pos = np.array([3.0, 0.0, -10])
    lookat = np.array([0.0, 0.0, 1.0])
    up = np.array([0, 1, 0])
    offset = [0, 0, 0]
    rotate = [-7.0, 25.0, 0.0]
    rotv = [0, 0, 0]
    fov = 90
    res = np.array([0, 0])
    rtrans = None
    
    @staticmethod
    def update_rtrans():
        global trans
        f = Util.normalize(Camera.lookat)
        u = Camera.up
        r = Util.cross_product(u, f)
        u = Util.cross_product(f, r)
        
        u = Util.normalize(u)
        r = Util.normalize(r)
        
        trans = np.matrix([
                          [r[0], u[0], f[0]],
                          [r[1], u[1], f[1]],
                          [r[2], u[2], f[2]]
                           ])
        rot = Util.get_rot_x_mat(Camera.rotate[0]) * Util.get_rot_y_mat(Camera.rotate[1])
        Camera.rtrans = trans * rot
        # TODO fill the result
class Render(object):
    '''
    classdocs
    '''
    image_texs = []  # image textures
    texs = []
    water_norms = []
    mtl_tex = None  # material texture
    mtl_num = 0
    trace_program = None  # program for tracing
    draw_program = None  # program to draw based on the result of trace_program
    fbo = None  # frame buffer object
    sample_count = 0
    time = -1
    time_start = Util.current_milli_time()
    elapsed = 0
    pause = False
    refresh = False

    def __init__(self, params):
        '''
        Constructor
        '''
        
    @staticmethod
    def get_shader(url, shader_type):
        file_content = open(url).read()
        shader = shaders.compileShader(file_content, shader_type)
        return shader

    @staticmethod
    def get_shader_program(vs_url, fs_url):
        '''Generate the GLSL program using vertex shader and fragment shader
        '''    
        vertex_shader = Render.get_shader(vs_url, GL_VERTEX_SHADER)
        fragment_shader = Render.get_shader(fs_url, GL_FRAGMENT_SHADER)
        program = shaders.compileProgram(vertex_shader, fragment_shader)
        return program
    
    @staticmethod
    def init_uniforms(loc_map):
        program = Render.trace_program
        for uniform, loc in loc_map.items():
            uniform_loc = glGetUniformLocation(program, uniform)
            if uniform_loc in [-1, None]:
                # raise Exception("Error %s" % uniform)
                print ("Error %s" % uniform)
                pass
            else:
                print("%s : %s" % (uniform, uniform_loc))
            setattr(program, loc, uniform_loc)
    
    @staticmethod
    def update_program():
        # c_float_p = ctypes.POINTER(ctypes.c_float)
        if not Render.pause:
            Render.elapsed = Util.current_milli_time() - Render.time_start
        else:
            # Render.sample_count = 0
            pass
        glUniform1i(getattr(Render.trace_program,
                            'stop_motion_loc'), 1 if Render.pause else 0)
        glUniform1i(getattr(Render.trace_program,
                            'refresh_loc'), 1 if Render.refresh else 0)
        glUniform1f(getattr(Render.trace_program,
                            'sample_count_loc'), Render.sample_count)
        glUniform1f(getattr(Render.trace_program,
                            'glob_time_loc'), Render.elapsed / 50)
        glUniform1f(getattr(Render.trace_program,
                            'cam_fov_loc'), math.tan(Util.radian(Camera.fov / 2)))
        glUniform2fv(getattr(Render.trace_program,
                            'cam_res_loc'), 1, np.array(Camera.res, 'f'))
        glUniform3fv(getattr(Render.trace_program,
                            'cam_pos_loc'), 1, np.array(Camera.pos, 'f'))
        glUniformMatrix3fv(getattr(Render.trace_program,
                            'cam_trans_loc'), 1, GL_FALSE, Util.flat(Camera.rtrans))
        Render.refresh = False
        # Util.flat(mat)
        
        
    @staticmethod
    def make_texture():
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, Camera.res[0], Camera.res[1], 0,
                     GL_RGB, GL_UNSIGNED_BYTE, None) 
        # glBindTexture(GL_TEXTURE_2D, None)
        return tex_id
    
    @staticmethod
    def make_texture_float(data):
        tex_id = glGenTextures(1)
        width = 4
        height = len(data) // width // 3
        # data_f = GLfloatArray(data)
        # data_f = array(data).tobytes()
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                     GL_RGB, GL_FLOAT, np.array(data, 'f'))
        # glBindTexture(GL_TEXTURE_2D, None)
        return (height - 1, tex_id)

    @staticmethod 
    def get_texture(image_src):
        ix, iy, image = Render.load_image(image_src)
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glGenerateMipmap(GL_TEXTURE_2D)
        # glBindTexture(GL_TEXTURE_2D, -1)
        return tex_id
        
        
    @staticmethod
    def load_image(imageName):
        im = pil_open(imageName)
        try:
            # # Note the conversion to RGB the crate bitmap is paletted!
            im = im.convert('RGB')
            ix, iy, image = im.size[0], im.size[1], im.tostring("raw", "RGBA", 0, -1)
        except SystemError:
            ix, iy, image = im.size[0], im.size[1], im.tostring("raw", "RGBX", 0, -1)
        assert ix * iy * 4 == len(image), """Image size != expected array size"""
        return ix, iy, image 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
