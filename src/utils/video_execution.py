import os
from moviepy.editor import VideoFileClip
from PIL import Image
import torch
import numpy as np
from decord import VideoReader, cpu, AudioReader

def split_video(video_file, start_time, end_time):
    """
    Split video into prefix part based on timestamp.
    video_file: path to video file
    start_time: start time in seconds
    end_time: end time in seconds
    """
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = os.path.join(os.path.dirname(video_file), "tmp")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")

    if os.path.exists(output_file):
        print(f"Video file {output_file} already exists.")
        return output_file

    video = VideoFileClip(video_file)

    video_duration = video.duration
    if start_time > video_duration:
        raise ValueError(f"Start time {start_time} exceeds the video duration {video_duration}")
    
    if end_time > video_duration:
        print(f"End time {end_time} exceeds video duration {video_duration}. Clipping to end of video.")
        end_time = video_duration
    clip = video.subclip(start_time, end_time)
    
    clip.write_videofile(output_file)
    clip.close()
    video.close()
    print(f"Video: {output_file} splitting completed.")
    return output_file
    




def _get_rawvideo_dec(
    video_path,
    image_processor,
    max_frames=32,
    min_frames=4,
    video_framerate=3,
    audio_segment_len=1.0,
    max_video_audio_segment=6,
    start_time=None,
    end_time=None,
    image_aspect_ratio="pad",
):
    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    video_time = len(vreader)/fps
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [
                all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)
            ]
        elif len(all_pos) < min_frames:
            # sample_pos = np.linspace(f_start, f_end, num=min_frames, dtype=int)
            # sample_pos = sample_pos.tolist()
            sample_pos = [
                all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)
            ]
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
        sample_time = [frame_id / fps for frame_id in sample_pos]

        if image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            patch_images = [
                expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean))
                for i in patch_images
            ]
            patch_images = [
                image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]
        else:
            patch_images = [
                image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]

        slice_len = len(patch_images)
        del vreader

        try:
            audio_reader = AudioReader(video_path, ctx=cpu(0), sample_rate=16000)
            audio_tensor = torch.from_numpy(audio_reader._array)
            del audio_reader
        except:
            audio_tensor = None

        if audio_tensor is None:
            return patch_images, slice_len, None, [0 for _ in range(slice_len)], sample_time, video_time
        else:
            # 自适应调整audio的长度, 但是仍然保持每段audio的长度是固定的，这样有较好的test time适应性
            num_audio_seg = round(((sample_pos[1] - sample_pos[0]) / fps) / audio_segment_len) # 防止0.99
            num_audio_seg = int(max(min(num_audio_seg, max_video_audio_segment), 1)) # [1, max_video_audio_segment]

            # 裁剪为audio片段
            # [frame][audio][audio][frame][audio][auido]
            audio_patch = []
            all_num_audio_seg = []
            for idx in sample_pos:
                start = int(max(idx / fps * 16000, 0))
                break_flag = 0
                for j in range(num_audio_seg):
                    end = int(start + 16000 * audio_segment_len)
                    if end > audio_tensor.shape[1]:
                        break_flag = 1
                        break
                    else:
                        audio_patch.append(audio_tensor[:, start: end])
                        start = end
                all_num_audio_seg.append(j if break_flag else j + 1)

            return patch_images, slice_len, audio_patch, all_num_audio_seg, sample_time, video_time
    else:
        print("video path: {} error.".format(video_path))
        raise FileNotFoundError