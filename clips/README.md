# VRS Clips

Put local MP4 clips here when testing the Docker Compose RTSP path.

The default Quick Start creates `falldown_test.mp4` automatically if it does not exist:

```bash
docker compose up --build
```

The clip is published as:

```text
rtsp://127.0.0.1:8554/falldown
http://127.0.0.1:8888/falldown/index.m3u8
```

## Use Your Own MP4

Copy or download an MP4 into this directory, then select it with `VRS_SAMPLE_CLIP`:

```bash
cp /path/to/site-camera-sample.mp4 clips/site-camera-sample.mp4
VRS_SAMPLE_CLIP=site-camera-sample.mp4 docker compose up --build
```

Open <http://127.0.0.1:5173/?tab=streams> to inspect the browser-playable HLS preview.

## What Compose Runs

The `rtsp-falldown` service publishes the selected MP4 into MediaMTX with an `ffmpeg` command equivalent to:

```bash
ffmpeg -re -stream_loop -1 -fflags +genpts \
  -i "clips/${VRS_SAMPLE_CLIP:-falldown_test.mp4}" \
  -map 0:v:0 -an \
  -vf fps=30,format=yuv420p \
  -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p \
  -f rtsp -rtsp_transport tcp \
  rtsp://127.0.0.1:8554/falldown
```

Inside Docker Compose, the input path is mounted as `/clips/...` and the RTSP target is `rtsp://rtsp:8554/falldown`. From the host, the same stream is available at `rtsp://127.0.0.1:8554/falldown`, and MediaMTX exposes the browser HLS preview at `http://127.0.0.1:8888/falldown/index.m3u8`.

## Download A Test Video

Use only videos that you are allowed to download and process.

Direct MP4 URL:

```bash
curl -L "https://example.com/path/to/video.mp4" -o clips/real-world-sample.mp4
VRS_SAMPLE_CLIP=real-world-sample.mp4 docker compose up --build
```

Video page supported by `yt-dlp`:

```bash
yt-dlp -f "bv*+ba/b" --merge-output-format mp4 \
  -o "clips/real-world-sample.mp4" \
  "https://example.com/video-page"

VRS_SAMPLE_CLIP=real-world-sample.mp4 docker compose up --build
```

The Compose stack only publishes the selected clip as RTSP/HLS. To generate fresh model-backed alerts for that video, run the inference profile after setting the same `VRS_SAMPLE_CLIP`.

```bash
VRS_SAMPLE_CLIP=real-world-sample.mp4 \
docker compose -f docker-compose.yaml -f docker-compose.hf-cache.yaml \
  --profile inference up --build
```
