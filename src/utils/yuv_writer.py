import numpy as np

from ..transforms.functional import rgb_to_ycbcr420


class YUVWriter():
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
        self.file = open(self.src_path, "wb")
        # pylint: enable=R1732

    def wirte_one_frame(self, img, src_format="rgb_to_yuv420"):
        '''
        Please note that the rgb converted here is different from that converted from ffmpeg.
        We suggest to use the rgb converted from ffmpeg as the input frame
        '''
        assert src_format in ["rgb_to_yuv420"]
        if src_format=="rgb_to_yuv420":
            width = img.shape[2]
            height = img.shape[3]
            y_size = width * height
            uv_size = width * height // 2
            img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            img_yuv420 = rgb_to_ycbcr420(img)
            img_y, img_uv = np.clip(np.rint(img_yuv420 * 255), 0, 255).astype(np.uint8)
            self.file.write(img_y.tobytes())
            self.file.write(img_uv.tobytes())

    def close(self):
        self.file.close()
