/* SPDX-FileCopyrightText: 2026 Toru Hashimoto
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * mp4frames: dump every Nth frame of a video as binary PPM.
 *
 * Exists because this machine has LichtFeld's FFmpeg 8 DLLs but no ffmpeg
 * executable, and the proxy requires auth so nothing can be downloaded.
 * Compiled locally against the vcpkg headers + import libs, so there is no
 * hand-maintained ABI to get wrong.
 *
 * Build (from a "x64 Native Tools" prompt, or after vcvars64.bat):
 *
 *   cl /nologo /O2 /W3 /I "%VCPKG%\include" mp4frames.c /Fe:mp4frames.exe ^
 *      /link /LIBPATH:"%VCPKG%\lib" avformat.lib avcodec.lib avutil.lib swscale.lib
 *
 * with VCPKG=D:\Apps\LichtFeld-Studio\build\vcpkg_installed\x64-windows.
 * Run with the DLL directory on PATH (D:\Apps\LichtFeld-Studio\build\Release).
 *
 * usage: mp4frames in.mp4 outdir every
 * writes outdir/frame_XXXXX.ppm where XXXXX is the 0-based frame index
 * (presentation order -- avcodec_receive_frame yields display order, which
 * is what novelview.py's keyframe alignment depends on).
 *
 * Exit status is 0 only when every expected frame was decoded AND written
 * intact: a missing outdir or a full disk must not look like success to a
 * scripted pipeline.
 */
#include <stdio.h>
#include <stdlib.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>

static struct SwsContext* g_sws = NULL;
static uint8_t* g_rgb[4] = {0};
static int g_rgb_ls[4] = {0};
static int g_w = 0, g_h = 0;

static int save_frame(const AVFrame* fr, const char* outdir, int index) {
    char path[1024];
    FILE* fp;
    int y;
    if (!g_sws) {
        g_w = fr->width;
        g_h = fr->height;
        g_sws = sws_getContext(g_w, g_h, (enum AVPixelFormat)fr->format,
                               g_w, g_h, AV_PIX_FMT_RGB24,
                               SWS_BILINEAR, NULL, NULL, NULL);
        if (!g_sws || av_image_alloc(g_rgb, g_rgb_ls, g_w, g_h,
                                     AV_PIX_FMT_RGB24, 1) < 0) {
            fprintf(stderr, "sws/alloc failed\n");
            return -1;
        }
    }
    sws_scale(g_sws, (const uint8_t* const*)fr->data, fr->linesize,
              0, g_h, g_rgb, g_rgb_ls);
    snprintf(path, sizeof(path), "%s/frame_%05d.ppm", outdir, index);
    fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "cannot write %s\n", path);
        return -1;
    }
    fprintf(fp, "P6\n%d %d\n255\n", g_w, g_h);
    for (y = 0; y < g_h; ++y)
        fwrite(g_rgb[0] + (size_t)y * g_rgb_ls[0], 1, (size_t)g_w * 3, fp);
    /* A truncated frame (disk full) must not count as saved. */
    if (ferror(fp)) {
        fprintf(stderr, "write error on %s\n", path);
        fclose(fp);
        return -1;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "close error on %s\n", path);
        return -1;
    }
    return 0;
}

int main(int argc, char** argv) {
    const char *in, *outdir;
    int every, vs, index = 0, saved = 0;
    AVFormatContext* fmt = NULL;
    const AVCodec* dec = NULL;
    AVCodecContext* ctx;
    AVPacket* pkt;
    AVFrame* fr;

    if (argc < 4) {
        fprintf(stderr, "usage: %s in.mp4 outdir every\n", argv[0]);
        return 2;
    }
    in = argv[1];
    outdir = argv[2];
    every = atoi(argv[3]);
    if (every < 1)
        every = 1;

    if (avformat_open_input(&fmt, in, NULL, NULL) < 0) {
        fprintf(stderr, "cannot open %s\n", in);
        return 1;
    }
    if (avformat_find_stream_info(fmt, NULL) < 0)
        return 1;
    vs = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &dec, 0);
    if (vs < 0 || !dec) {
        fprintf(stderr, "no video stream/decoder\n");
        return 1;
    }
    ctx = avcodec_alloc_context3(dec);
    if (avcodec_parameters_to_context(ctx, fmt->streams[vs]->codecpar) < 0)
        return 1;
    if (avcodec_open2(ctx, dec, NULL) < 0) {
        fprintf(stderr, "decoder open failed\n");
        return 1;
    }
    pkt = av_packet_alloc();
    fr = av_frame_alloc();

    while (av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index == vs && avcodec_send_packet(ctx, pkt) == 0) {
            while (avcodec_receive_frame(ctx, fr) == 0) {
                if (index % every == 0 && save_frame(fr, outdir, index) == 0)
                    ++saved;
                ++index;
            }
        }
        av_packet_unref(pkt);
    }
    avcodec_send_packet(ctx, NULL);          /* drain */
    while (avcodec_receive_frame(ctx, fr) == 0) {
        if (index % every == 0 && save_frame(fr, outdir, index) == 0)
            ++saved;
        ++index;
    }
    printf("decoded %d frames, saved %d\n", index, saved);
    /* Success requires every EXPECTED save to have happened, not merely
     * some decoding: with a missing outdir every fopen fails and a
     * "decoded > 0" test would still report success. */
    return (index > 0 && saved == (index + every - 1) / every) ? 0 : 1;
}
