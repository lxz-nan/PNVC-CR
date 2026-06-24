import numpy as np

from ..transforms.functional import ycbcr420_to_rgb


class YUVReader():
    def __init__(self, src_path, width, height, src_format='420'):
        self.src_path = src_path + '/' + src_path.split('/')[-1] + '.yuv'
        self.width = width
        self.height = height
        self.src_format = src_format
        self.y_size = width * height
        if src_format == '420':
            self.uv_size = width * height // 2
        else:
            assert False
        self.eof = False
        # pylint: disable=R1732
        self.file = open(self.src_path, "rb")
        # pylint: enable=R1732

    def read_one_frame(self, src_format="420"):
        '''
        Please note that the rgb converted here is different from that converted from ffmpeg.
        We suggest to use the rgb converted from ffmpeg as the input frame
        '''
        assert src_format in ["420", "rgb", "both"]

        def _none_exist_frame():
            if src_format == "420":
                return None, None
            if src_format == "rgb":
                return None
            return None, None, None
        if self.eof:
            return _none_exist_frame()
        y = self.file.read(self.y_size)
        uv = self.file.read(self.uv_size)
        if not y or not uv:
            self.eof = True
            return _none_exist_frame()
        
        y = np.frombuffer(y, dtype=np.uint8).copy().reshape(1, self.height, self.width)
        uv = np.frombuffer(uv, dtype=np.uint8).copy().reshape(2, self.height // 2, self.width // 2)
        y = y.astype(np.float32) / 255
        uv = uv.astype(np.float32) / 255
        if src_format == "420":
            return y, uv

        rgb = ycbcr420_to_rgb(y, uv, order=0)
        if src_format == "rgb":
            return rgb
        return y, uv, rgb

    def close(self):
        self.file.close()
