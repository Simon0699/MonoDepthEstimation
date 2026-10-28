from torchcodec.decoders import VideoDecoder
import torchvision.transforms.functional as f
import torch
def process_video(video_url, write_file):
    decoder = VideoDecoder(video_url, num_ffmpeg_threads = 0)
    lend = len(decoder)
    i = 0
    avg_fps = (decoder.metadata.average_fps)
    target_fps = 10
    skip_size = round(avg_fps/target_fps)
    CHUNK_SIZE = 150 * skip_size
    assert(skip_size > 0)
    chunks = []
    while (i < lend):
        end = min(i+CHUNK_SIZE,lend)
        chunk = decoder[i:end:skip_size]
        chunk = f.resize(chunk, (224,224))
        chunks.append(chunk)
        i += CHUNK_SIZE
    torch.save(torch.cat(chunks,0), write_file) ## stores as uint8 without dividng by 255 for memory efficiency