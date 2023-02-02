import pdb, shutil
import PIL
from PIL import Image
from glob import glob
import numpy as np
import cv2
import os
osp = os.path
from math import ceil

"""triplets:
- light / violin / tennis
- oranges
- 2 bears / (bull)
- pain / (discord) / mii - faceless
- (statue) / mlisa / chatgpt
- (pokemon) / ssb / geico - gaming
- (jojo) / sharpener / stapler
- viewfinder / smartphone / (magic ball) - technology
- (gcal) / ? / ?
- mri / mri2 / printer - buildings
- corky / pot / wallet - Disney cartoon
- apt windows / (gcal) / ? - grid
remaining - nascar, vatican, fireworks, plasma, moai, monster, dio
"""
small_rows = ['light', 'violin',
        'orange2', 'orange1',
        'ssb_sq', 'side_hustle',
        
        'mri2', 'mri3', #printer siege
        'nascar', 'vatican',
         #'wa1', 'viewfinder', 

        'mlisa', 'chatgpt', 
        'sharpener', 'stapler',
        'bear', 'shocked_bear',
        'pain', 'mii', 

        'mdew', 'abstract',
        'pot', 'wallet',
        'moai1', 'monster',
        'apt_windows', 'room',
        #'plasma', 'water', 'duck'
        None, None
]

big_rows = ['tennis', 'orange4', 'char', 
'mri', 'magic_ball',
'statue', 'jojo', 'bull', 'discord',
'mdew1', 'fireworks', 'corky', 'gcal', None] #'dio',

class ImageWrapper:
    def __init__(self, name, size):
        if name is None:
            self.imgs = None
            return
        elif name in os.listdir('collage'):
            self.imgs = [Image.open(os.path.join('collage', name, file)) \
                for file in sorted(os.listdir(os.path.join('collage', name)))]
            self.frame_rate = 25
            self.prev_img = None
            self.loop = True
            self.init_pause = 1
        elif name+".png" in os.listdir('collage/still'): # single image
            self.imgs = [Image.open(os.path.join('collage', 'still', name+".png"))]
            self.frame_rate = 0
            self.init_pause = 0
        elif name+".mp4" in os.listdir('collage/video_loop'):
            self.imgs = self.open_mp4(os.path.join('collage/video_loop', name+".mp4"))
            self.frame_rate = 1
            self.loop = True
            self.init_pause = 4 # frames to wait for the video to come into view
        elif name+".mp4" in os.listdir('collage/vid_no_loop'):
            self.imgs = self.open_mp4(os.path.join('collage/vid_no_loop', name+".mp4"))
            self.frame_rate = 1
            self.loop = False
            self.init_pause = 4
        else:
            raise ValueError("Image not found: " + name)

        self.resize(size)
        self.ix = 0

    def resize(self, size):
        imgs = []
        for img in self.imgs:
            if img.size[0] == size:
                imgs = self.imgs
                break
            elif img.size[0] < size:
                imgs.append(img.resize((size,size), Image.Resampling.BICUBIC))
            else:
                imgs.append(img.resize((size,size), Image.Resampling.LANCZOS))
        self.imgs = [img.convert('RGB') for img in imgs]
            
    def open_mp4(self, path):
        imgs = []
        cap = cv2.VideoCapture(path)
        while(cap.isOpened()):
            ret,cv2_im = cap.read()
            if ret:
                converted = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
                imgs.append(Image.fromarray(converted))
            else:
                break
        return imgs

    def next(self):
        if self.frame_rate == 0: # still image
            img = self.imgs[0]
        else:
            if self.init_pause > 0:
                self.init_pause -= 1
                return np.array(self.imgs[0])
            if self.ix == len(self.imgs * self.frame_rate):
                if self.loop:
                    self.ix = 0
                else:
                    self.ix -= 1
            if self.frame_rate == 1:
                img = self.imgs[self.ix]
            elif self.frame_rate > 1:
                img = self.imgs[self.ix//self.frame_rate]
                if self.ix % self.frame_rate == 0 and self.prev_img is not None:
                    img = (np.array(self.prev_img)/2 + np.array(img)/2).astype('uint8')
                self.prev_img = img
            self.ix += 1
        return np.array(img)

class ImageConcat:
    def __init__(self, im1,im2, delay=True):
        self.im1 = im1
        if im2.imgs is None:
            self.im2 = None
            return
        if delay and im2.init_pause > 0:
            im2.init_pause += 124 # 384/speed
        self.im2 = im2

    def next(self):
        if self.im2 is None:
            print('should not be possible')
            return
        return np.concatenate((self.im1.next(), self.im2.next()), axis=0)


class Context:
    def __init__(self):
        self.right_bot = self.load_small_images(small_rows[:2], delay=False)
        self.left_bot = self.load_big_image(big_rows[0])
        self.moving = True
    
    def load_small_images(self, paths, delay=True):
        return ImageConcat(ImageWrapper(paths[0], 384),
            ImageWrapper(paths[1], 384), delay=delay)
    def load_big_image(self, path):
        return ImageWrapper(path, 768)

    def get_image_for_frame(self, row):
        if (offset := row % 768) == 0 and self.moving:
            self.left_top = self.left_bot
            self.right_top = self.right_bot
            if (ix := row // 768) % 2 == 1:
                self.right_bot = self.load_small_images(small_rows[2*(ix+1):2*(ix+2)])
                self.left_bot = self.load_big_image(big_rows[ix+1])
            else:
                self.left_bot = self.load_small_images(small_rows[2*(ix+1):2*(ix+2)])
                self.right_bot = self.load_big_image(big_rows[ix+1])
        
        top = np.concatenate((self.left_top.next(), self.right_top.next()), axis=1)[offset:]
        if offset == 0:
            return top

        bot = np.concatenate((self.left_bot.next(), self.right_bot.next()), axis=1)[:offset-768]
        return np.concatenate((top, bot), axis=0)

if __name__ == '__main__':
    # for folder in os.listdir('collage'):
    #     if folder.startswith('vid'):
    #         continue
    #     for path in os.listdir(os.path.join('collage', folder)):
    #         if not path.endswith('png') or '_thumb.' in path:
    #             os.remove(os.path.join('collage', folder, path))
    #             continue
    #         img = Image.open(os.path.join('collage', folder, path))
    #         if img.size != (768, 768):
    #             w,h = img.size
    #             if w > h:
    #                 dx = w-h
    #                 img = img.crop((dx//2, 0, w-((dx+1)//2), h))
    #                 img.save(os.path.join('collage', folder, path))
    #             elif w < h:
    #                 dx = h-w
    #                 img = img.crop((0, dx//2, w, h-((dx+1)//2)))
    #                 img.save(os.path.join('collage', folder, path))
    # exit()

    n_rows = len(big_rows)
    last_row = (n_rows-1) * 768
    speed = 4
    out_folder = 'coll_out'
    # shutil.rmtree(out_folder, ignore_errors=True)
    os.makedirs(out_folder, exist_ok=True)
    context = Context()
    top = frame = 0
    init_pause = 5 #50
    while top < last_row:
        img = context.get_image_for_frame(top)
        Image.fromarray(img).save(osp.join(out_folder, f'{frame:04d}.png'))
        if init_pause > 0:
            context.moving = False
            init_pause -= 1
        else:
            context.moving = True
            top += speed
        frame += 1

    assert top == last_row
    for _ in range(20):
        img = context.get_image_for_frame(top)
        Image.fromarray(img).save(osp.join(out_folder, f'{frame:04d}.png'))
        context.moving = False
        frame += 1
