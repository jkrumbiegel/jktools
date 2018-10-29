from moviepy.editor import VideoClip
import numpy as np


class RGBAVideoClip(VideoClip):

    def __init__(self, make_rgba_frame, duration=None):

        self.rgba_buffer = None
        self.last_t = None

        def save_last_rgba_frame(t):
            # only create a new frame if the time is different from that of the last frame
            if t != self.last_t:
                self.rgba_buffer = make_rgba_frame(t)
                if not isinstance(self.rgba_buffer, np.ndarray):
                    raise Exception(f'The rgba buffer is not a numpy array but of type "{type(self.rgba_buffer)}".')
                if self.rgba_buffer.dtype != np.uint8:
                    raise Exception(f'The rgba buffer needs to be an 8-bit uint array, not "{self.rgba_buffer.dtype}".')

                # update the time stamp of the last created frame
                self.last_t = t

        # frame function for image data
        def make_frame(t):
            save_last_rgba_frame(t)
            return self.rgba_buffer[..., :3]

        # frame function for mask data
        def make_mask_frame(t):
            save_last_rgba_frame(t)
            return self.rgba_buffer[..., 3] / 255

        super(RGBAVideoClip, self).__init__(make_frame, duration=duration)

        self.mask = VideoClip(make_mask_frame, ismask=True, duration=duration)
