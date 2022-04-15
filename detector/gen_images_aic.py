import os
import sys
sys.path.append('../')
from config import cfg
import cv2
from tqdm import tqdm

def preprocess(src_root, dst_root):
    if not os.path.isdir(src_root):
        print(f"[Err]: invalid source root: {src_root}")
        return

    if not os.path.isdir(dst_root):
        os.makedirs(dst_root)
        print("{} made".format(dst_root))

    dst_dir = dst_root + '/images/test/'
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)        

    if os.path.isdir(src_root):
        scene_list = os.listdir(src_root)
        for y in scene_list:
            if y.startswith('S'):
                print(f"Preprocessing {y}")
                y_path = os.path.join(src_root,y)
                cam_list = os.listdir(y_path)
                for zi in tqdm(range(len(cam_list))):
                    z = cam_list[zi]
                    z_path = os.path.join(y_path,z)
                    if z.startswith('c'):
                        video_path = os.path.join(z_path,'vdo.avi')
                        roi_path = os.path.join(z_path, 'roi.jpg')
                        ignor_region = cv2.imread(roi_path)

                        dst_img1_dir = os.path.join(dst_dir,y,z,'img1')
                        if not os.path.isdir(dst_img1_dir):
                            os.makedirs(dst_img1_dir)

                        video = cv2.VideoCapture(video_path)
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        frame_current = 0
                        while frame_current<frame_count-1:
                            frame_current = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                            _, frame = video.read()
                            dst_f =  'img{:06d}.jpg'.format(frame_current)
                            dst_f_path = os.path.join(dst_img1_dir , dst_f)
                            if not os.path.isfile(dst_f_path):
                                frame = draw_ignore_regions(frame, ignor_region)
                                cv2.imwrite(dst_f_path, frame)
                                #print('{}:{} generated to {}'.format(z,dst_f, dst_img1_dir))
                            else:
                                pass
                                #print('{}:{} already exists.'.format(z,dst_f))

def draw_ignore_regions(img, region):
    if img is None:
        print('[Err]: Input image is none!')
        return -1
    img = img*(region>0)

    return img

if __name__ == '__main__':
    cfg.merge_from_file(f'../config/{sys.argv[1]}')
    cfg.freeze()
    save_dir = cfg.CHALLENGE_DATA_DIR
    preprocess(src_root=f'{cfg.CHALLENGE_DATA_DIR}',
               dst_root=f'{save_dir}')
    print('Done')
