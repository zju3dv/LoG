from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from imgui_bundle import imgui
    from easyvolcap.runners.volumetric_video_viewer import VolumetricVideoViewer

import glm
import torch
import numpy as np
from typing import List, Optional

from os.path import join
from scipy import interpolate
from copy import copy, deepcopy

from glm import vec2, vec3, vec4, mat3, mat4, mat4x3, mat2x3  # This is actually highly optimized

from .camera import Camera

class CameraPath:
    # This is the Model in the EVC gui designs

    # Basic a list of cameras with interpolations
    # Use the underlying Camera class as backbone
    # Will export to a sequence of cameras extri.yml and intri.yml
    # Will support keyframes based manipulations:
    # 1. Adding current view as key frame
    # 2. Jumping to previous keyframe (resize window as well?) (toggle edit state?) (or just has a replace button)
    #    - Snap to current keyframe?
    #    - Editing would be much better (just a button to replace the selected keyframe)
    # 3. Toggle playing animation of this keyframe (supports some degress of control)
    # 4. Export the animation as a pair of extri.yml and intri.yml
    # 5. imguizmo control of the created camera in the list (translation, rotation etc)
    def __init__(self,
                 playing: bool = False,
                 playing_time: float = 0.5,
                 playing_speed: float = 0.0005,

                 n_render_views: int = 100,
                 render_plots: bool = True,

                 # Visualization related
                 visible: bool = True,
                 name: str = 'camera_path',
                 filename: str = '',
                 plot_thickness: float = 8.0,
                 camera_thickness: float = 6.0,
                 plot_color: int = 0x80ff80ff,
                 camera_color: int = 0x80ffffff,
                 camera_axis_size: float = 0.10,

                 **kwargs,
                 ) -> None:
        self.keyframes: List[Camera] = []  # orders matter
        self.playing_time = playing_time  # range: 0-1

        self.playing = playing  # is this playing? update cam if it is
        self.playing_speed = playing_speed  # faster interpolation time
        self.n_render_views = n_render_views
        self.render_plots = render_plots

        # Private
        self.cursor_index = -1  # the camera to edit
        self.periodic = True

        # Visualization
        self.name = name
        self.visible = visible
        self.plot_thickness = plot_thickness
        self.camera_thickness = camera_thickness
        self.plot_color = plot_color
        self.camera_color = camera_color
        self.camera_axis_size = camera_axis_size
        if filename:
            self.load_keyframes(filename)

    def __len__(self):
        return len(self.keyframes)

    @property
    def loop_interp(self):
        return self.periodic

    @loop_interp.setter
    def loop_interp(self, v: bool):
        changed = self.periodic != v
        self.periodic = v
        if changed: self.update()  # only perform heavy operation after change

    @property
    def selected(self):
        return self.cursor_index

    @selected.setter
    def selected(self, v: int):
        if v >= len(self): return
        if not len(self): self.cursor_index = -1; return
        self.cursor_index = range(len(self))[v]
        denom = (len(self) - 1)
        if denom: self.playing_time = self.cursor_index / denom  # 1 means last frame
        else: self.playing_time = 0.5

    def replace(self, camera: Camera):
        self.keyframes[self.selected] = deepcopy(camera)
        self.update()

    def insert(self, camera: Camera):
        self.keyframes = self.keyframes[:self.selected + 1] + [deepcopy(camera)] + self.keyframes[self.selected + 1:]
        self.selected = self.selected + 1
        self.update()

    def delete(self, index: int):
        del self.keyframes[index]
        self.selected = self.selected - 1  # go back one
        self.update()

    def clear(self):
        self.keyframes.clear()
        self.selected = -1

    def update(self):
        # MARK: HEAVY
        K = len(self.keyframes)
        if K <= 3: return

        # Prepare for linear and extrinsic parameters
        ks = np.asarray([c.K.to_list() for c in self.keyframes]).transpose(0, 2, 1).reshape(K, -1)  # 9
        hs = np.asarray([c.H for c in self.keyframes]).reshape(K, -1)
        ws = np.asarray([c.W for c in self.keyframes]).reshape(K, -1)
        ns = np.asarray([c.n for c in self.keyframes]).reshape(K, -1)
        fs = np.asarray([c.f for c in self.keyframes]).reshape(K, -1)
        ts = np.asarray([c.t for c in self.keyframes]).reshape(K, -1)
        vs = np.asarray([c.v for c in self.keyframes]).reshape(K, -1)
        bs = np.asarray([c.bounds.to_list() for c in self.keyframes]).reshape(K, -1)  # 6
        lins = np.concatenate([ks, hs, ws, ns, fs, ts, vs, bs], axis=-1)  # K, D
        c2ws = np.asarray([c.c2w.to_list() for c in self.keyframes]).transpose(0, 2, 1)  # K, 3, 4

        # Recompute interpolation parameters
        self.lin_func = gen_linear_interp_func(lins, smoothing_term=0.0 if self.periodic else 10.0)  # smoothness: 0 -> period, >0 -> non-period, -1 orbit (not here)
        self.c2w_func = gen_cubic_spline_interp_func(c2ws, smoothing_term=0.0 if self.periodic else 10.0)

    def interp(self, us: float, **kwargs):
        K = len(self.keyframes)
        if K <= 3: return

        # MARK: HEAVY?
        # Actual interpolation
        lin = self.lin_func(us)
        c2w = self.c2w_func(us)

        # Extract linear parameters
        K = torch.as_tensor(lin[:9]).view(3, 3)  # need a transpose
        H = int(lin[9])
        W = int(lin[10])
        n = torch.as_tensor(lin[11], dtype=torch.float)
        f = torch.as_tensor(lin[12], dtype=torch.float)
        t = torch.as_tensor(lin[13], dtype=torch.float)
        v = torch.as_tensor(lin[14], dtype=torch.float)
        bounds = torch.as_tensor(lin[15:]).view(2, 3)  # no need for transpose

        # Extract splined parameters
        w2c = affine_inverse(torch.as_tensor(c2w))  # already float32
        R = w2c[:3, :3]
        T = w2c[:3, 3:]

        return H, W, K, R, T, n, f, t, v, bounds

    def export_keyframes(self, path: str):
        # Store keyframes to path
        cameras = {f'{i:06d}': k.to_easymocap() for i, k in enumerate(self.keyframes)}
        write_camera(cameras, path)  # without extri.yml, only dirname
        log(yellow(f'Keyframes saved to: {blue(path)}'))

    def load_keyframes(self, path: str):
        # Store keyframes to path
        cameras = read_camera(join(path, 'intri.yml'), join(path, 'extri.yml'))
        cameras = dotdict({k: cameras[k] for k in sorted(cameras.keys())})  # assuming dict is ordered (python 3.7+)
        self.keyframes = [Camera().from_easymocap(cam) for cam in cameras.values()]
        self.name = path
        self.update()

    def export_interps(self, path: str):
        # Store interpolations (animation) to path
        us = np.linspace(0, 1, self.n_render_views, dtype=np.float32)

        cameras = dotdict()
        for i, u in enumerate(tqdm(us, desc='Exporting interpolated cameras')):
            cameras[f'{i:06d}'] = self.interp(u).to_easymocap()
        write_camera(cameras, path)  # without extri.yml, only dirname
        log(yellow(f'Interpolated cameras saved to: {blue(path)}'))

    def render_imgui(self, viewer: 'VolumetricVideoViewer', batch: dotdict):
        # from easyvolcap.utils.gl_utils import Mesh
        # Mesh.render_imgui(self, viewer, batch)
        from imgui_bundle import imgui
        from easyvolcap.utils.imgui_utils import push_button_color, pop_button_color, col2rgba, col2vec4, vec42col, list2col, col2imu32

        i = batch.i
        will_delete = batch.will_delete
        slider_width = batch.slider_width

        imgui.push_item_width(slider_width * 0.5)
        self.name = imgui.input_text(f'Mesh name##{i}', self.name)[1]
        self.n_render_views = imgui.slider_int(f'Plot samples##{i}', self.n_render_views, 0, 3000)[1]
        self.plot_thickness = imgui.slider_float(f'Plot thickness##{i}', self.plot_thickness, 0.01, 10.0)[1]
        self.camera_thickness = imgui.slider_float(f'Camera thickness##{i}', self.camera_thickness, 0.01, 10.0)[1]
        self.camera_axis_size = imgui.slider_float(f'Camera axis size##{i}', self.camera_axis_size, 0.01, 1.0)[1]

        self.plot_color = list2col(imgui.color_edit4(f'Plot color##{i}', col2vec4(self.plot_color), flags=imgui.ColorEditFlags_.no_inputs.value)[1])
        self.camera_color = list2col(imgui.color_edit4(f'Camera color##{i}', col2vec4(self.camera_color), flags=imgui.ColorEditFlags_.no_inputs.value)[1])

        push_button_color(0x55cc33ff if not self.render_plots else 0x8855aaff)
        if imgui.button(f'No Plot##{i}' if not self.render_plots else f' Plot ##{i}'):
            self.render_plots = not self.render_plots
        pop_button_color()

        imgui.same_line()
        push_button_color(0x55cc33ff if not self.visible else 0x8855aaff)
        if imgui.button(f'Show##{i}' if not self.visible else f'Hide##{i}'):
            self.visible = not self.visible
        pop_button_color()

        # Render the delete button
        imgui.same_line()
        push_button_color(0xff5533ff)
        if imgui.button(f'Delete##{i}'):
            will_delete.append(i)
        pop_button_color()

        # The actual rendering
        self.draw(viewer.camera)

    def draw(self, camera: Camera):

        # The actual rendering starts here, the camera paths are considered GUI elements for eaiser management
        # This rendering pattern is extremly slow and hard on the CPU, but whatever for now, just visualization
        if not self.visible: return
        if not len(self): return
        proj = camera.w2p  # 3, 4

        # Render cameras
        for i, cam in enumerate(self.keyframes):
            ixt = cam.ixt
            c2w = cam.c2w
            c2w = mat4x3(c2w)  # vis cam only supports this

            # Add to imgui rendering list
            visualize_cameras(proj, ixt, c2w, col=self.camera_color, thickness=self.camera_thickness, axis_size=self.camera_axis_size)

        if self.render_plots and len(self) >= 4:
            us = np.linspace(0, 1, self.n_render_views, dtype=np.float32)
            c2ws = self.c2w_func(us)
            cs = c2ws[..., :3, 3]  # N, 3
            for i, c in enumerate(cs):
                if i == 0:
                    p = c  # previous
                    continue
                add_debug_line(proj, vec3(*p), vec3(*c), col=self.plot_color, thickness=self.plot_thickness)
                p = c

    def render(self, camera: Camera):
        pass



if __name__ == "__main__":
    pass