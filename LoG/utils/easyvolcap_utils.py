import torch
import numpy as np
from imgui_bundle import imgui_color_text_edit as ed
from imgui_bundle import portable_file_dialogs as pfd
from imgui_bundle import imgui, imguizmo, imgui_toggle, immvision, implot, ImVec2, ImVec4, imgui_md, immapp, hello_imgui

from LoG.dataset.base import prepare_camera
from LoG.utils.trainer import prepare_batch

from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import log, red
from easyvolcap.utils.timer_utils import timer
from easyvolcap.utils.imgui_utils import push_button_color, pop_button_color
from easyvolcap.utils.data_utils import to_cuda, to_numpy, add_batch, Visualization
from easyvolcap.utils.viewer_utils import Camera, CameraPath, visualize_cameras, visualize_cube, add_debug_line, add_debug_text, visualize_axes, add_debug_text_2d

from easyvolcap.engine import cfg
from easyvolcap.utils.data_utils import load_image

import glfw
from easyvolcap.runners.volumetric_video_viewer import VolumetricVideoViewer

class Viewer(VolumetricVideoViewer):
    def __init__(self,
                 window_size = [1080, 1920],  # height, width
                 window_title: str = f'LoG',  # MARK: global config
                 exp_name: str = 'random',
                 font_size: int = 64,
                 font_bold: str = 'submodules/EasyVolcap/assets/fonts/CascadiaCodePL-Bold.otf',
                 font_italic: str = 'submodules/EasyVolcap/assets/fonts/CascadiaCodePL-Italic.otf',
                 font_default: str = 'submodules/EasyVolcap/assets/fonts/CascadiaCodePL-Regular.otf',
                 icon_file: str = 'submodules/EasyVolcap/assets/imgs/easyvolcap.png',

                 use_window_focal: bool = True,
                 use_quad_cuda: bool = True,
                 use_quad_draw: bool = False,

                 update_fps_time: float = 0.5,  # be less stressful
                 update_mem_time: float = 0.5,  # be less stressful

                 skip_exception: bool = False,  # always pause to give user a debugger
                 compose: bool = False,
                 compose_power: float = 1.0,
                 render_meshes: bool = True,
                 render_network: bool = True,

                 mesh_preloading = [],
                 splat_preloading = [],
                 show_preloading: bool = True,

                 fullscreen: bool = False,
                 camera_cfg: dotdict = dotdict(type=Camera.__name__, string='{"H":2032,"W":3840,"K":[[4279.6650390625,0.0,1920.0],[0.0,4279.6650390625,992.4420776367188],[0.0,0.0,1.0]],"R":[[0.41155678033828735,0.911384105682373,0.0],[-0.8666263818740845,0.39134538173675537,0.3095237910747528],[0.2820950746536255,-0.12738661468029022,0.9508903622627258]],"T":[[-4.033830642700195],[-1.7978200912475586],[3.9347341060638428]],"n":0.10000000149011612,"f":1000.0,"t":0.0,"v":0.0,"bounds":[[-10.0,-10.0,-3.0],[10.0,10.0,4.0]],"mass":0.10000000149011612,"moment_of_inertia":0.10000000149011612,"movement_force":1.0,"movement_torque":1.0,"movement_speed":5.0,"origin":[0.0,0.0,0.0],"world_up":[0.0,0.0,-1.0]}'),

                 
                 

                 show_metrics_window: bool = False,
                 show_demo_window: bool = False,

                 visualize_axes: bool = False,  # will add an extra 0.xms
                 ):
        # Camera related configurations
        self.camera_cfg = camera_cfg
        self.fullscreen = fullscreen
        self.window_size = window_size
        self.window_title = window_title
        self.use_window_focal = use_window_focal

        # Quad related configurations
        self.use_quad_draw = use_quad_draw
        self.use_quad_cuda = use_quad_cuda
        self.compose = compose  # composing only works with cudagl for now
        self.compose_power = compose_power

        # Font related config
        self.font_default = font_default
        self.font_italic = font_italic
        self.font_bold = font_bold
        self.font_size = font_size
        self.icon_file = icon_file

        self.render_meshes = render_meshes
        self.render_network = render_network

        self.update_fps_time = update_fps_time
        self.update_mem_time = update_mem_time

        self.exposure = 1.0
        self.offset = 0.0

        self.init_camera(camera_cfg)  # prepare for the actual rendering now, needs dataset -> needs runner
        self.init_glfw()  # ?: this will open up the window and let the user wait, should we move this up?
        self.init_imgui()

        from easyvolcap.engine import args
        args.type = 'gui'  # manually setting this parameter
        self.init_opengl()
        self.init_quad()
        self.bind_callbacks()

        from easyvolcap.utils.gl_utils import Mesh, Splat

        self.meshes: List[Mesh] = [
            *[Mesh(filename=mesh, visible=show_preloading, render_normal=True) for mesh in mesh_preloading],
            *[Splat(filename=splat, visible=show_preloading, point_radius=0.0015, H=self.H, W=self.W) for splat in splat_preloading],
        ]

        self.camera_path = CameraPath()
        self.visualize_axes = visualize_axes
        self.visualize_paths = True
        self.visualize_cameras = True
        self.visualize_bounds = True
        self.epoch = 0
        self.runner = dotdict(ep_iter=0, collect_timing=False, timer_record_to_file=False, timer_sync_cuda=False)
        self.use_vsync = True
        self.dataset = dotdict()
        self.visualization_type = Visualization.RENDER
        self.playing = False
        self.discrete_t = False
        self.playing_speed = 0.0
        self.network_available = False

        # Initialize other parameters
        self.show_demo_window = show_demo_window
        self.show_metrics_window = show_metrics_window

        # Others
        self.skip_exception = skip_exception
        self.static = dotdict(batch=dotdict(), output=dotdict())  # static data store updated through the rendering
        self.dynamic = dotdict()

    def init_camera(self, camera_cfg: dotdict):
        self.camera = Camera(**camera_cfg)
        self.camera.front = self.camera.front  # perform alignment correction

    def frame(self):
        # print(f'framing: {time.perf_counter()}')
        import OpenGL.GL as gl
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.dynamic = dotdict()

        # Render GS
        if self.render_network:
            batch = easyvolcap_camera_to_fastnb_camera(self.camera.to_batch())
            with torch.no_grad():
                output = self.renderer.vis(dotdict(camera=batch), self.model)
            image = output['render'][0].permute(1, 2, 0)
            if self.exposure != 1.0 or self.offset != 0.0:
                image = torch.cat([(image[..., :3] * self.exposure + self.offset), image[..., -1:]], dim=-1)  # add manual correction
            image = (image.clip(0, 1) * 255).type(torch.uint8).flip(0)  # transform

            self.quad.copy_to_texture(image)
            self.quad.draw()  # draw is typically faster by 0.5ms

        # Render meshes (or point clouds)
        if self.render_meshes:
            for mesh in self.meshes:
                mesh.render(self.camera)

        self.draw_imgui()  # defines GUI elements
        self.show_imgui()

    def draw_rendering_gui(self, batch: dotdict = dotdict(), output: dotdict = dotdict()):

        # Other rendering options like visualization type
        if imgui.collapsing_header('Rendering'):
            self.visualize_axes = imgui_toggle.toggle('Visualize axes', self.visualize_axes, config=self.static.toggle_ios_style)[1]
            self.visualize_bounds = imgui_toggle.toggle('Visualize bounds', self.visualize_bounds, config=self.static.toggle_ios_style)[1]
            self.visualize_cameras = imgui_toggle.toggle('Visualize cameras', self.visualize_cameras, config=self.static.toggle_ios_style)[1]

    def draw_imgui(self):
        from easyvolcap.utils.gl_utils import Mesh, Splat, Gaussian

        # Initialization
        glfw.poll_events()  # process pending events, keyboard and stuff
        imgui.backends.opengl3_new_frame()
        imgui.backends.glfw_new_frame()
        imgui.new_frame()
        imgui.push_font(self.default_font)

        self.static.playing_time = self.camera_path.playing_time  # Remember this, if changed, update camera
        self.static.slider_width = imgui.get_window_width() * 0.65  # https://github.com/ocornut/imgui/issues/267
        self.static.toggle_ios_style = imgui_toggle.ios_style(size_scale=0.2)

        # Titles
        fps, frame_time = self.get_fps_and_frame_time()
        name, device, memory = self.get_device_and_memory()
        # glfw.set_window_title(self.window, self.window_title.format(FPS=fps)) # might confuse window managers
        self.static.fps = fps
        self.static.frame_time = frame_time
        self.static.name = name
        self.static.device = device
        self.static.memory = memory

        # Being the main window
        imgui.begin(f'{self.W}x{self.H} {fps:.3f} fps###main', flags=imgui.WindowFlags_.menu_bar)

        self.draw_menu_gui()
        self.draw_banner_gui()
        self.draw_camera_gui()
        self.draw_rendering_gui()
        self.draw_keyframes_gui()
        # self.draw_model_gui()
        self.draw_mesh_gui()
        self.draw_debug_gui()

        # End of main window and rendering
        imgui.end()

        imgui.pop_font()
        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

    def glfw_scroll_callback(self, window, x_offset, y_offset):
        if (imgui.get_io().want_capture_mouse):
            return imgui.backends.glfw_scroll_callback(self.window_address, x_offset, y_offset)
        CONTROL = glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
        SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        ALT = glfw.get_key(window, glfw.KEY_LEFT_ALT) == glfw.PRESS

        if CONTROL and SHIFT:
            self.camera.fx = self.camera.fx + y_offset * 50  # locked xy
        elif CONTROL:
            self.camera.n = max(self.camera.n + y_offset * 0.01, 0.001)
        elif SHIFT:
            self.camera.f = self.camera.f + y_offset * 0.1
        elif ALT:
            self.render_ratio = self.render_ratio + y_offset * 0.01
            self.render_ratio = min(max(self.render_ratio, 0.05), 1.0)
        else:
            self.camera.move(x_offset, y_offset)

    def glfw_mouse_button_callback(self, window, button, action, mods):
        # Let the UI handle its corrsponding operations
        if (imgui.get_io().want_capture_mouse):
            imgui.backends.glfw_mouse_button_callback(self.window_address, button, action, mods)
            if action != glfw.RELEASE:
                return  # only return if not releasing the mouse

        x, y = glfw.get_cursor_pos(window)
        if (action == glfw.PRESS or action == glfw.REPEAT):
            SHIFT = mods & glfw.MOD_SHIFT
            CONTROL = mods & glfw.MOD_CONTROL
            MIDDLE = button == glfw.MOUSE_BUTTON_MIDDLE
            LEFT = button == glfw.MOUSE_BUTTON_LEFT
            RIGHT = button == glfw.MOUSE_BUTTON_RIGHT

            is_panning = SHIFT or RIGHT
            about_origin = LEFT or (MIDDLE and SHIFT)
            self.camera.begin_dragging(x, y, is_panning, about_origin)
        elif action == glfw.RELEASE:
            self.camera.end_dragging()
        else:
            pass
            # log(red('Mouse button callback falling through'), button)

    def show_imgui(self):
        # Update and Render additional Platform Windows
        # (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this code elsewhere.
        #  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
        io = imgui.get_io()
        if io.config_flags & imgui.ConfigFlags_.viewports_enable > 0:
            backup_current_context = glfw.get_current_context()
            imgui.update_platform_windows()
            imgui.render_platform_windows_default()
            glfw.make_context_current(backup_current_context)
        glfw.swap_buffers(self.window)

    def init_glfw(self):
        if not glfw.init():
            log(red('Could not initialize OpenGL context'))
            exit(1)

        # Decide GL+GLSL versions
        # GL 3.3 + GLSL 330
        self.glsl_version = '#version 330'
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)  # // 3.2+ only
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, 1)  # 1 is gl.GL_TRUE

        # Create a windowed mode window and its OpenGL context
        window = glfw.create_window(self.W, self.H, self.window_title, None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)  # disable vsync

        icon = load_image(self.icon_file)
        pixels = (icon * 255).astype(np.uint8)
        height, width = icon.shape[:2]
        glfw.set_window_icon(window, 1, [width, height, pixels])  # set icon for the window

        if not window:
            glfw.terminate()
            log(red('Could not initialize window'))
            raise RuntimeError('Failed to initialize window in glfw')

        self.window = window
        cfg.window = window  # MARK: GLOBAL VARIABLE

def easyvolcap_camera_to_fastnb_camera(batch: dotdict):
    K = batch.K
    R = batch.R  # 3, 3
    T = batch.T  # 3, 1
    H = batch.H
    W = batch.W
    C = -R.mT @ T  # 3, 1

    cam = dotdict(H=H, W=W, K=K, R=R, T=T, center=C)
    cam = prepare_camera(to_numpy(cam), 1, batch.n, batch.f)
    return add_batch(to_cuda(cam))

