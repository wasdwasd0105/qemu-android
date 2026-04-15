/*
 * QEMU AAudio audio backend for Android
 *
 * Uses the Android AAudio NDK C API in data-callback mode so that the
 * hardware buffer is always fully populated — any gap between what the
 * guest frontend produces and what AAudio pulls is filled with silence.
 * This mirrors the SDL, CoreAudio and PipeWire backends and fixes the
 * "repeated-tone on pause" artifact with ac97, whose BUP_LAST underrun
 * path emits the last sample repeated rather than zeros.
 *
 * Copyright (c) 2025
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/module.h"
#include "audio.h"

#include <pthread.h>

#define AUDIO_CAP "aaudio"
#include "audio_int.h"

#include <aaudio/AAudio.h>
#ifdef __ANDROID__
#include <android/log.h>
#endif

/*
 * Diagnostic knobs.  Flip to 1 in a debug build to trace audio-path issues
 * like the ac97-pause repeated-tone.  Leave at 0 in production — both add
 * work or noise to the realtime audio callback.
 *
 * AAUDIO_DEBUG_TRACE
 *   Classify each output callback's ring contents as one of
 *     UNDERRUN — zero-filled because the ring was empty
 *     ZERO    — ring held only zero frames
 *     DC      — every 32-bit frame equal and non-zero (matches ac97
 *               BUP_LAST, which repeats last_samp)
 *     VARYING — real audio
 *   Logs once per state change (and on DC-value change) so you get one
 *   line on pause/unpause, not per-callback spam.  Assumes 2- or 4-byte
 *   alignment; accurate for S16-stereo and F32-mono, rough for F32-stereo.
 *
 * AAUDIO_FORCE_SILENCE
 *   Ignore the ring entirely and always emit zeros from aaudio_out_cb.
 *   If noise still comes through with this on, it is NOT coming from
 *   the QEMU frontend or this backend's ring — suspect routing, the
 *   Android audio HAL, or the device speaker path.
 *
 * On Android the diagnostic lines go straight to logcat (tag "aaudio")
 * via __android_log_print, bypassing the stderr pipe — this is what the
 * rest of the port uses (see system/android_entry.c) and works reliably
 * from the realtime audio callback thread.  Other platforms fall back to
 * stderr.
 */
#define AAUDIO_DEBUG_TRACE    0
#define AAUDIO_FORCE_SILENCE  0

#ifdef __ANDROID__
#define AA_TRACE(...) \
    __android_log_print(ANDROID_LOG_INFO, "aaudio", __VA_ARGS__)
#else
#define AA_TRACE(...) \
    do { fprintf(stderr, "aaudio: " __VA_ARGS__); \
         fputc('\n', stderr); fflush(stderr); } while (0)
#endif

typedef struct AAudioVoiceOut {
    HWVoiceOut hw;
    pthread_mutex_t mutex;
    AAudioStream *stream;
} AAudioVoiceOut;

typedef struct AAudioVoiceIn {
    HWVoiceIn hw;
    pthread_mutex_t mutex;
    AAudioStream *stream;
} AAudioVoiceIn;

static aaudio_format_t qemu_to_aaudio_fmt(AudioFormat fmt)
{
    switch (fmt) {
    case AUDIO_FORMAT_S16:
        return AAUDIO_FORMAT_PCM_I16;
    case AUDIO_FORMAT_F32:
        return AAUDIO_FORMAT_PCM_FLOAT;
    default:
        /* Not natively supported — ask AAudio for S16 and let mixeng
         * convert the guest's format to/from S16 on the QEMU side. */
        return AAUDIO_FORMAT_PCM_I16;
    }
}

static AudioFormat aaudio_to_qemu_fmt(aaudio_format_t fmt)
{
    switch (fmt) {
    case AAUDIO_FORMAT_PCM_I16:
        return AUDIO_FORMAT_S16;
    case AAUDIO_FORMAT_PCM_FLOAT:
        return AUDIO_FORMAT_F32;
    default:
        dolog("Unexpected AAudio format %d, assuming S16\n", fmt);
        return AUDIO_FORMAT_S16;
    }
}

/* ---- Data callbacks (run on AAudio's high-priority audio thread) ---- */

static aaudio_data_callback_result_t aaudio_out_cb(AAudioStream *stream,
                                                   void *userdata,
                                                   void *audioData,
                                                   int32_t numFrames)
{
    HWVoiceOut *hw = userdata;
    AAudioVoiceOut *aa = (AAudioVoiceOut *)hw;
    size_t want = (size_t)numFrames * hw->info.bytes_per_frame;
    uint8_t *out = audioData;

#if AAUDIO_DEBUG_TRACE
    {
        /* One-shot heartbeat: proves the callback is being invoked at all,
         * independent of any state-change filtering below. */
        static int seen_first;
        if (!seen_first) {
            seen_first = 1;
            AA_TRACE("out_cb first invocation numFrames=%d bpf=%d",
                     numFrames, hw->info.bytes_per_frame);
        }
    }
#endif

#if AAUDIO_FORCE_SILENCE
    /* Diagnostic: bypass the ring buffer entirely.  If you still hear noise
     * with this enabled, the problem is downstream of this callback. */
    audio_pcm_info_clear_buf(&hw->info, audioData, numFrames);
    return AAUDIO_CALLBACK_RESULT_CONTINUE;
#endif

    pthread_mutex_lock(&aa->mutex);
    while (hw->pending_emul && want) {
        size_t start = audio_ring_posb(hw->pos_emul, hw->pending_emul,
                                       hw->size_emul);
        size_t chunk = MIN(MIN(hw->pending_emul, want),
                           hw->size_emul - start);

        memcpy(out, (uint8_t *)hw->buf_emul + start, chunk);
        hw->pending_emul -= chunk;
        out += chunk;
        want -= chunk;
    }
    pthread_mutex_unlock(&aa->mutex);

#if AAUDIO_DEBUG_TRACE
    /* Classify what the ring just handed us so we can distinguish a real
     * underrun (ring empty) from the frontend feeding repeated samples
     * (ac97 BUP_LAST case).  Only log on state change to avoid spam. */
    {
        size_t total = (size_t)numFrames * hw->info.bytes_per_frame;
        size_t real_bytes = total - want;
        static int prev_state = -1;
        static uint32_t prev_dc;
        int state;
        uint32_t dc = 0;

        if (real_bytes < sizeof(uint32_t)) {
            state = 0; /* UNDERRUN */
        } else {
            const uint32_t *p = audioData;
            size_t n = real_bytes / sizeof(uint32_t);
            uint32_t first = p[0];
            bool all_zero = true;
            bool all_same = true;

            for (size_t i = 0; i < n; i++) {
                if (p[i] != 0)     all_zero = false;
                if (p[i] != first) all_same = false;
                if (!all_zero && !all_same) {
                    break;
                }
            }

            if (all_zero) {
                state = 1; /* ZERO */
            } else if (all_same) {
                state = 2; /* DC / BUP_LAST */
                dc = first;
            } else {
                state = 3; /* VARYING */
            }
        }

        if (state != prev_state || (state == 2 && dc != prev_dc)) {
            static const char * const names[] = {
                "UNDERRUN (zero-filled)",
                "ZERO (ring held zeros)",
                "DC (BUP_LAST-like)",
                "VARYING (real audio)",
            };

            if (state == 2) {
                AA_TRACE("out_cb state: %s value=0x%08x",
                         names[state], dc);
            } else {
                AA_TRACE("out_cb state: %s", names[state]);
            }
            prev_state = state;
            prev_dc = dc;
        }
    }
#endif

    /* Underrun: fill the rest with real silence rather than letting AAudio
     * replay the previous hardware buffer or passing on whatever the
     * frontend synthesises (ac97 BUP_LAST repeats last_samp). */
    if (want) {
        audio_pcm_info_clear_buf(&hw->info, out,
                                 want / hw->info.bytes_per_frame);
    }
    return AAUDIO_CALLBACK_RESULT_CONTINUE;
}

static aaudio_data_callback_result_t aaudio_in_cb(AAudioStream *stream,
                                                  void *userdata,
                                                  void *audioData,
                                                  int32_t numFrames)
{
    HWVoiceIn *hw = userdata;
    AAudioVoiceIn *aa = (AAudioVoiceIn *)hw;
    size_t have = (size_t)numFrames * hw->info.bytes_per_frame;
    uint8_t *in = audioData;

    pthread_mutex_lock(&aa->mutex);
    while (hw->pending_emul < hw->size_emul && have) {
        size_t chunk = MIN(have,
                           MIN(hw->size_emul - hw->pos_emul,
                               hw->size_emul - hw->pending_emul));

        memcpy((uint8_t *)hw->buf_emul + hw->pos_emul, in, chunk);
        hw->pending_emul += chunk;
        hw->pos_emul = (hw->pos_emul + chunk) % hw->size_emul;
        in += chunk;
        have -= chunk;
    }
    pthread_mutex_unlock(&aa->mutex);
    /* If the ring is full we silently drop — matches sdl_callback_in. */
    return AAUDIO_CALLBACK_RESULT_CONTINUE;
}

/* ---- Stream open ---- */

static AAudioStream *aaudio_open_stream(struct audsettings *as,
                                        aaudio_direction_t direction,
                                        AAudioStream_dataCallback cb,
                                        void *userdata)
{
    AAudioStreamBuilder *builder = NULL;
    AAudioStream *stream = NULL;
    aaudio_result_t res;
    int32_t burst;

    res = AAudio_createStreamBuilder(&builder);
    if (res != AAUDIO_OK) {
        dolog("AAudio_createStreamBuilder failed: %d\n", res);
        return NULL;
    }

    AAudioStreamBuilder_setDirection(builder, direction);
    AAudioStreamBuilder_setSampleRate(builder, as->freq);
    AAudioStreamBuilder_setChannelCount(builder, as->nchannels);
    AAudioStreamBuilder_setFormat(builder, qemu_to_aaudio_fmt(as->fmt));
    AAudioStreamBuilder_setPerformanceMode(builder,
                                           AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
    AAudioStreamBuilder_setSharingMode(builder, AAUDIO_SHARING_MODE_SHARED);
    AAudioStreamBuilder_setDataCallback(builder, cb, userdata);

#if __ANDROID_API__ >= 28
    /* Tell Android what the stream is for. Without this, playback may be
     * routed to the earpiece instead of the speaker and the media volume
     * key may not control it. */
    if (direction == AAUDIO_DIRECTION_OUTPUT) {
        AAudioStreamBuilder_setUsage(builder, AAUDIO_USAGE_MEDIA);
        AAudioStreamBuilder_setContentType(builder,
                                           AAUDIO_CONTENT_TYPE_MUSIC);
    } else {
        AAudioStreamBuilder_setInputPreset(builder,
                                           AAUDIO_INPUT_PRESET_GENERIC);
    }
#endif

    res = AAudioStreamBuilder_openStream(builder, &stream);
    AAudioStreamBuilder_delete(builder);

    if (res != AAUDIO_OK) {
        dolog("AAudioStreamBuilder_openStream failed: %d\n", res);
        return NULL;
    }

    /* AAudio prefers buffer size to be a multiple of the device's burst
     * size. 2× burst is the sweet spot for low-latency shared streams —
     * capacity (allocated by AAudio) is typically 4–8× burst. */
    burst = AAudioStream_getFramesPerBurst(stream);
    if (burst > 0) {
        AAudioStream_setBufferSizeInFrames(stream, burst * 2);
    }

    return stream;
}

/* ---- Playback ---- */

static int aaudio_init_out(HWVoiceOut *hw, struct audsettings *as,
                           void *drv_opaque)
{
    AAudioVoiceOut *aa = (AAudioVoiceOut *)hw;
    struct audsettings obt_as;
    int32_t burst;

    pthread_mutex_init(&aa->mutex, NULL);

    aa->stream = aaudio_open_stream(as, AAUDIO_DIRECTION_OUTPUT,
                                    aaudio_out_cb, hw);
    if (!aa->stream) {
        pthread_mutex_destroy(&aa->mutex);
        return -1;
    }

#if AAUDIO_DEBUG_TRACE
    AA_TRACE("init_out: stream=%p requested freq=%d ch=%d fmt=%d",
             aa->stream, as->freq, as->nchannels, as->fmt);
#endif

    /* Use what AAudio actually gave us; mixeng converts between the guest
     * format and this on the QEMU side. Requested != obtained on many
     * devices (most commonly channel count). */
    obt_as = *as;
    obt_as.freq      = AAudioStream_getSampleRate(aa->stream);
    obt_as.nchannels = AAudioStream_getChannelCount(aa->stream);
    obt_as.fmt       = aaudio_to_qemu_fmt(AAudioStream_getFormat(aa->stream));
    obt_as.endianness = 0; /* AAudio PCM is host-endian */

    audio_pcm_init_info(&hw->info, &obt_as);

    /* QEMU-side ring: enough to bridge one timer_period of frontend writes
     * across many callback pulls. 8× burst ≈ 32 ms at a typical 192-frame
     * burst @ 48 kHz — comfortably larger than the default timer period. */
    burst = AAudioStream_getFramesPerBurst(aa->stream);
    hw->samples = burst > 0 ? burst * 8
                            : AAudioStream_getBufferCapacityInFrames(aa->stream);

    return 0;
}

static void aaudio_fini_out(HWVoiceOut *hw)
{
    AAudioVoiceOut *aa = (AAudioVoiceOut *)hw;

    if (aa->stream) {
        AAudioStream_requestStop(aa->stream);
        AAudioStream_close(aa->stream);
        aa->stream = NULL;
    }
    pthread_mutex_destroy(&aa->mutex);
}

static void aaudio_enable_out(HWVoiceOut *hw, bool enable)
{
    AAudioVoiceOut *aa = (AAudioVoiceOut *)hw;

    if (!aa->stream) {
        return;
    }

    if (enable) {
        AAudioStream_requestStart(aa->stream);
    } else {
        AAudioStream_requestPause(aa->stream);
    }
}

/* Locked wrappers around the generic ring-buffer helpers. The AAudio
 * callback thread reads/writes the same fields. */
#define AA_WRAP_OUT(name, ret_t, args_decl, args)                       \
    static ret_t aaudio_##name args_decl                                \
    {                                                                   \
        AAudioVoiceOut *aa = (AAudioVoiceOut *)hw;                      \
        ret_t r;                                                        \
        pthread_mutex_lock(&aa->mutex);                                 \
        r = audio_generic_##name args;                                  \
        pthread_mutex_unlock(&aa->mutex);                               \
        return r;                                                       \
    }

AA_WRAP_OUT(buffer_get_free, size_t, (HWVoiceOut *hw), (hw))
AA_WRAP_OUT(get_buffer_out, void *, (HWVoiceOut *hw, size_t *size),
            (hw, size))
AA_WRAP_OUT(put_buffer_out, size_t,
            (HWVoiceOut *hw, void *buf, size_t size),
            (hw, buf, size))
AA_WRAP_OUT(write, size_t, (HWVoiceOut *hw, void *buf, size_t size),
            (hw, buf, size))

/* ---- Capture ---- */

static int aaudio_init_in(HWVoiceIn *hw, struct audsettings *as,
                          void *drv_opaque)
{
    AAudioVoiceIn *aa = (AAudioVoiceIn *)hw;
    struct audsettings obt_as;
    int32_t burst;

    pthread_mutex_init(&aa->mutex, NULL);

    aa->stream = aaudio_open_stream(as, AAUDIO_DIRECTION_INPUT,
                                    aaudio_in_cb, hw);
    if (!aa->stream) {
        pthread_mutex_destroy(&aa->mutex);
        return -1;
    }

    obt_as = *as;
    obt_as.freq      = AAudioStream_getSampleRate(aa->stream);
    obt_as.nchannels = AAudioStream_getChannelCount(aa->stream);
    obt_as.fmt       = aaudio_to_qemu_fmt(AAudioStream_getFormat(aa->stream));
    obt_as.endianness = 0;

    audio_pcm_init_info(&hw->info, &obt_as);

    burst = AAudioStream_getFramesPerBurst(aa->stream);
    hw->samples = burst > 0 ? burst * 8
                            : AAudioStream_getBufferCapacityInFrames(aa->stream);

    /* The capture callback writes into hw->buf_emul directly, so it must
     * exist before the stream is started. The output side relies on
     * audio_generic_get_buffer_out to allocate lazily. */
    hw->size_emul = hw->samples * hw->info.bytes_per_frame;
    hw->buf_emul = g_malloc(hw->size_emul);
    hw->pos_emul = hw->pending_emul = 0;

    return 0;
}

static void aaudio_fini_in(HWVoiceIn *hw)
{
    AAudioVoiceIn *aa = (AAudioVoiceIn *)hw;

    if (aa->stream) {
        AAudioStream_requestStop(aa->stream);
        AAudioStream_close(aa->stream);
        aa->stream = NULL;
    }
    pthread_mutex_destroy(&aa->mutex);
}

static void aaudio_enable_in(HWVoiceIn *hw, bool enable)
{
    AAudioVoiceIn *aa = (AAudioVoiceIn *)hw;

    if (!aa->stream) {
        return;
    }

    if (enable) {
        AAudioStream_requestStart(aa->stream);
    } else {
        AAudioStream_requestStop(aa->stream);
    }
}

#define AA_WRAP_IN(name, ret_t, args_decl, args)                        \
    static ret_t aaudio_##name args_decl                                \
    {                                                                   \
        AAudioVoiceIn *aa = (AAudioVoiceIn *)hw;                        \
        ret_t r;                                                        \
        pthread_mutex_lock(&aa->mutex);                                 \
        r = audio_generic_##name args;                                  \
        pthread_mutex_unlock(&aa->mutex);                               \
        return r;                                                       \
    }

AA_WRAP_IN(read, size_t, (HWVoiceIn *hw, void *buf, size_t size),
           (hw, buf, size))
AA_WRAP_IN(get_buffer_in, void *, (HWVoiceIn *hw, size_t *size),
           (hw, size))

static void aaudio_put_buffer_in(HWVoiceIn *hw, void *buf, size_t size)
{
    AAudioVoiceIn *aa = (AAudioVoiceIn *)hw;

    pthread_mutex_lock(&aa->mutex);
    audio_generic_put_buffer_in(hw, buf, size);
    pthread_mutex_unlock(&aa->mutex);
}

/* ---- Driver ---- */

static void *aaudio_audio_init(Audiodev *dev, Error **errp)
{
    assert(dev->driver == AUDIODEV_DRIVER_AAUDIO);
    return dev;
}

static void aaudio_audio_fini(void *opaque)
{
}

static struct audio_pcm_ops aaudio_pcm_ops = {
    .init_out        = aaudio_init_out,
    .fini_out        = aaudio_fini_out,
    .write           = aaudio_write,
    .buffer_get_free = aaudio_buffer_get_free,
    .get_buffer_out  = aaudio_get_buffer_out,
    .put_buffer_out  = aaudio_put_buffer_out,
    .enable_out      = aaudio_enable_out,

    .init_in         = aaudio_init_in,
    .fini_in         = aaudio_fini_in,
    .read            = aaudio_read,
    .get_buffer_in   = aaudio_get_buffer_in,
    .put_buffer_in   = aaudio_put_buffer_in,
    .enable_in       = aaudio_enable_in,
};

static struct audio_driver aaudio_audio_driver = {
    .name           = "aaudio",
    .descr          = "Android AAudio audio",
    .init           = aaudio_audio_init,
    .fini           = aaudio_audio_fini,
    .pcm_ops        = &aaudio_pcm_ops,
    .max_voices_out = 1,
    .max_voices_in  = 1,
    .voice_size_out = sizeof(AAudioVoiceOut),
    .voice_size_in  = sizeof(AAudioVoiceIn),
};

static void register_audio_aaudio(void)
{
    audio_driver_register(&aaudio_audio_driver);
}
type_init(register_audio_aaudio);
