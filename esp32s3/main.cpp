#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_http_server.h"
#include "nvs_flash.h"

#include "esp_camera.h"
#include "img_converters.h"

#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"

#include "test_image_0000_128x96_rgb.h"

static const char *TAG = "DEPTH_V1";

// ============================================================
// Embedded ESP-DL model
// ============================================================

extern const uint8_t model_espdl[]
    asm("_binary_student_boundary_v1_128x96_esp32s3_espdl_start");

extern const uint8_t model_espdl_end[]
    asm("_binary_student_boundary_v1_128x96_esp32s3_espdl_end");

// ============================================================
// Depth configuration
// ============================================================

#define DEPTH_WIDTH   128
#define DEPTH_HEIGHT   96
#define DEPTH_SIZE    (DEPTH_WIDTH * DEPTH_HEIGHT)
#define MODEL_RGB_SIZE (DEPTH_SIZE * 3)

// ============================================================
// XIAO ESP32-S3 Sense camera pins
// ============================================================

// Known-working XIAO ESP32-S3 Sense OV2640 pin mapping
// Taken from the user's previously working camera project.
#define CAM_PIN_PWDN    21
#define CAM_PIN_RESET   1
#define CAM_PIN_XCLK    10
#define CAM_PIN_SIOD    40
#define CAM_PIN_SIOC    39

#define CAM_PIN_D7      48
#define CAM_PIN_D6      11
#define CAM_PIN_D5      12
#define CAM_PIN_D4      14
#define CAM_PIN_D3      16
#define CAM_PIN_D2      18
#define CAM_PIN_D1      17
#define CAM_PIN_D0      15

#define CAM_PIN_VSYNC   38
#define CAM_PIN_HREF    47
#define CAM_PIN_PCLK    13

// Camera stream resolution.
// JPEG is used so the browser can display the live image directly.
#define CAMERA_WIDTH    320
#define CAMERA_HEIGHT   240

// Persistent depth buffer: 12 KB.
static int8_t *g_depth_data = nullptr;

// PSRAM RGB888 workspace used to convert/resize camera frames.
static uint8_t *g_camera_rgb = nullptr;

// -----------------------------------------------------------------
// Shared published frame.
// The camera JPEG and depth map are published together only after
// inference finishes. This guarantees that /camera and /depth refer
// to the same captured camera frame.
// -----------------------------------------------------------------
#define MAX_JPEG_SIZE (80 * 1024)

static uint8_t *g_pending_jpeg = nullptr;
static size_t g_pending_jpeg_len = 0;

static uint8_t *g_latest_jpeg = nullptr;
static size_t g_latest_jpeg_len = 0;

static SemaphoreHandle_t g_publish_mutex = nullptr;
static bool g_frame_ready = false;

// 128x96 RGB input buffer for ESP-DL.
// 36,864 bytes, allocated in PSRAM.
static float *g_model_input_float = nullptr;

static int g_depth_exponent = -7;
static float g_inference_ms = 0.0f;
static float g_fps = 0.0f;
static float g_min_depth = 0.0f;
static float g_max_depth = 0.0f;
static float g_mean_depth = 0.0f;

static volatile bool g_inference_busy = false;

// ============================================================
// Memory information
// ============================================================

static void print_memory(const char *label)
{
    ESP_LOGI(TAG, "%s", label);

    ESP_LOGI(TAG,
             "  Internal free : %u KB",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));

    ESP_LOGI(TAG,
             "  PSRAM free    : %u KB",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));

    ESP_LOGI(TAG,
             "  Internal largest block : %u KB",
             (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) / 1024));

    ESP_LOGI(TAG,
             "  PSRAM largest block    : %u KB",
             (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) / 1024));
}

// ============================================================
// Camera initialization
// ============================================================

static bool camera_init()
{
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "==============================================");
    ESP_LOGI(TAG, "Initializing XIAO ESP32-S3 Sense camera");
    ESP_LOGI(TAG, "==============================================");

    camera_config_t config = {};

    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;

    config.pin_d0 = CAM_PIN_D0;
    config.pin_d1 = CAM_PIN_D1;
    config.pin_d2 = CAM_PIN_D2;
    config.pin_d3 = CAM_PIN_D3;
    config.pin_d4 = CAM_PIN_D4;
    config.pin_d5 = CAM_PIN_D5;
    config.pin_d6 = CAM_PIN_D6;
    config.pin_d7 = CAM_PIN_D7;

    config.pin_xclk = CAM_PIN_XCLK;
    config.pin_pclk = CAM_PIN_PCLK;
    config.pin_vsync = CAM_PIN_VSYNC;
    config.pin_href = CAM_PIN_HREF;
    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;
    config.pin_pwdn = CAM_PIN_PWDN;
    config.pin_reset = CAM_PIN_RESET;

    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    // QVGA is a good compromise between image quality and memory.
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_LATEST;

    esp_err_t err = esp_camera_init(&config);

    if (err != ESP_OK)
    {
        ESP_LOGE(TAG,
                 "Camera init failed: 0x%x (%s)",
                 err,
                 esp_err_to_name(err));
        return false;
    }

    sensor_t *s = esp_camera_sensor_get();

    if (s != nullptr)
    {
        // Keep the camera image reasonably natural.
        s->set_brightness(s, 0);
        s->set_contrast(s, 0);
        s->set_saturation(s, 0);
    }

    ESP_LOGI(TAG, "Camera initialized successfully.");
    ESP_LOGI(TAG, "Camera output: %dx%d JPEG",
             CAMERA_WIDTH, CAMERA_HEIGHT);

    return true;
}

// ============================================================
// Wi-Fi Access Point
// ============================================================

static void wifi_init_ap()
{
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "==============================================");
    ESP_LOGI(TAG, "Starting Wi-Fi Access Point");
    ESP_LOGI(TAG, "==============================================");

    esp_err_t ret = nvs_flash_init();

    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }

    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_netif_init());

    ESP_ERROR_CHECK(
        esp_event_loop_create_default()
    );

    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();

    ESP_ERROR_CHECK(
        esp_wifi_init(&cfg)
    );

    wifi_config_t wifi_config = {};

    strcpy((char *)wifi_config.ap.ssid, "DEPTH_V1_ESP32S3");
    strcpy((char *)wifi_config.ap.password, "depth123");

    wifi_config.ap.ssid_len =
        strlen("DEPTH_V1_ESP32S3");

    wifi_config.ap.channel = 1;
    wifi_config.ap.max_connection = 4;
    wifi_config.ap.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(
        esp_wifi_set_mode(WIFI_MODE_AP)
    );

    ESP_ERROR_CHECK(
        esp_wifi_set_config(
            WIFI_IF_AP,
            &wifi_config
        )
    );

    ESP_ERROR_CHECK(
        esp_wifi_start()
    );

    ESP_LOGI(TAG, "Wi-Fi AP started.");
    ESP_LOGI(TAG, "SSID     : DEPTH_V1_ESP32S3");
    ESP_LOGI(TAG, "Password : depth123");
    ESP_LOGI(TAG, "IP       : 192.168.4.1");
}

// ============================================================
// HTTP: Main webpage
// ============================================================

static esp_err_t http_root_handler(httpd_req_t *req)
{
    const char *html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ESP32-S3 Depth V1</title>
<style>
body {
    background:#111;
    color:white;
    font-family:Arial,sans-serif;
    text-align:center;
    margin:16px;
}
h1 { margin-bottom:16px; }

.container {
    display:flex;
    justify-content:center;
    align-items:flex-start;
    gap:18px;
    flex-wrap:wrap;
}

.panel {
    display:flex;
    flex-direction:column;
    align-items:center;
}

.panel-title {
    font-size:18px;
    margin-bottom:7px;
}

img, canvas {
    width:512px;
    height:384px;
    image-rendering:pixelated;
    border:2px solid white;
    object-fit:contain;
    background:#000;
}

.info {
    margin-top:16px;
    font-size:16px;
    line-height:1.6;
}

#status { font-weight:bold; }

@media (max-width:1100px) {
    img, canvas {
        width:90vw;
        height:auto;
    }
}
</style>
</head>

<body>

<h1>Depth Anything V2 — ESP32-S3</h1>

<div class="container">

    <div class="panel">
        <div class="panel-title">Live RGB Camera</div>
        <img id="cameraImage" alt="Live camera">
    </div>

    <div class="panel">
        <div class="panel-title">Depth Colormap</div>
        <canvas id="colorCanvas"
                width="128"
                height="96"></canvas>
    </div>

</div>

<div class="info">
    <div id="status">Connecting...</div>
    <div id="stats"></div>
</div>

<script>

const cameraImage =
    document.getElementById("cameraImage");

const colorCanvas =
    document.getElementById("colorCanvas");

const colorCtx =
    colorCanvas.getContext("2d");

const status =
    document.getElementById("status");

const stats =
    document.getElementById("stats");

let updateBusy = false;

function depthColor(v)
{
    let r, g, b;

    // Far -> blue
    // Middle -> green/yellow
    // Near -> red

    if (v < 0.25) {
        r = 0;
        g = Math.round(v * 4 * 255);
        b = 255;
    }
    else if (v < 0.50) {
        r = 0;
        g = 255;
        b = Math.round((0.50 - v) * 4 * 255);
    }
    else if (v < 0.75) {
        r = Math.round((v - 0.50) * 4 * 255);
        g = 255;
        b = 0;
    }
    else {
        r = 255;
        g = Math.round((1.0 - v) * 4 * 255);
        b = 0;
    }

    return [r, g, b];
}

function updateCamera()
{
    // Timestamp prevents browser caching.
    cameraImage.src =
        "/camera?t=" + Date.now();
}

async function updateDepth()
{
    if (updateBusy)
        return;

    updateBusy = true;

    try {

        const response =
            await fetch("/depth?t=" + Date.now(), {
                cache: "no-store"
            });

        if (!response.ok)
            throw new Error("HTTP " + response.status);

        const buffer =
            await response.arrayBuffer();

        if (buffer.byteLength < 16)
            throw new Error(
                "Response too small: " +
                buffer.byteLength
            );

        const view =
            new DataView(buffer);

        const magic =
            String.fromCharCode(
                view.getUint8(0),
                view.getUint8(1),
                view.getUint8(2),
                view.getUint8(3)
            );

        if (magic !== "DEP1")
            throw new Error("Invalid depth packet");

        const width =
            view.getUint16(4, true);

        const height =
            view.getUint16(6, true);

        const exponent =
            view.getInt8(8);

        const payloadSize =
            view.getUint32(12, true);

        if (width !== 128 || height !== 96)
            throw new Error(
                "Unexpected resolution"
            );

        if (payloadSize !== width * height)
            throw new Error(
                "Invalid payload size"
            );

        if (buffer.byteLength <
            16 + payloadSize)
        {
            throw new Error(
                "Incomplete depth packet"
            );
        }

        const depth =
            new Int8Array(
                buffer,
                16,
                payloadSize
            );

        let rawMin = 127;
        let rawMax = -128;

        for (let i = 0; i < depth.length; i++)
        {
            const q = depth[i];

            if (q < rawMin)
                rawMin = q;

            if (q > rawMax)
                rawMax = q;
        }

        const scale =
            Math.pow(2, exponent);

        const minDepth =
            rawMin * scale;

        const maxDepth =
            rawMax * scale;

        const range =
            rawMax - rawMin;

        let sum = 0;

        const colorImage =
            colorCtx.createImageData(
                width,
                height
            );

        for (let i = 0; i < depth.length; i++)
        {
            const q = depth[i];

            let normalized =
                range > 0
                ? (q - rawMin) / range
                : 0.0;

            normalized =
                Math.max(
                    0,
                    Math.min(1, normalized)
                );

            const c =
                depthColor(normalized);

            colorImage.data[i * 4 + 0] = c[0];
            colorImage.data[i * 4 + 1] = c[1];
            colorImage.data[i * 4 + 2] = c[2];
            colorImage.data[i * 4 + 3] = 255;

            sum += q * scale;
        }

        const meanDepth =
            sum / depth.length;

        colorCtx.putImageData(
            colorImage,
            0,
            0
        );

        status.textContent =
            "ESP32-S3 live camera + depth: OK";

        stats.innerHTML =
            "Depth: " + width + " × " + height +
            " | INT8 exponent: " + exponent +
            " | Min: " + minDepth.toFixed(4) +
            " | Max: " + maxDepth.toFixed(4) +
            " | Mean: " + meanDepth.toFixed(4) +
            " | Inference: " +
            (window.lastInferenceMs || "see serial") +
            " ms" +
            " | Payload: " + payloadSize +
            " bytes";

    }
    catch (error)
    {
        status.textContent =
            "Connection error: " + error;

        console.error(error);
    }
    finally
    {
        updateBusy = false;
    }
}

function update()
{
    updateCamera();
    updateDepth();
}

update();

setInterval(update, 700);

</script>

</body>
</html>
)rawliteral";

    httpd_resp_set_type(req, "text/html");
    httpd_resp_set_hdr(
        req,
        "Cache-Control",
        "no-cache, no-store"
    );

    return httpd_resp_send(
        req,
        html,
        HTTPD_RESP_USE_STRLEN
    );
}

// ============================================================
// HTTP: Live camera JPEG
// ============================================================

static esp_err_t http_camera_handler(httpd_req_t *req)
{
    if (g_publish_mutex == nullptr)
    {
        httpd_resp_send_err(
            req,
            HTTPD_500_INTERNAL_SERVER_ERROR,
            "Frame buffer unavailable");
        return ESP_FAIL;
    }

    if (xSemaphoreTake(g_publish_mutex, pdMS_TO_TICKS(2000)) != pdTRUE)
    {
        httpd_resp_send_err(
            req,
            HTTPD_500_INTERNAL_SERVER_ERROR,
            "Frame temporarily busy");
        return ESP_FAIL;
    }

    if (!g_frame_ready || g_latest_jpeg_len == 0)
    {
        xSemaphoreGive(g_publish_mutex);

        httpd_resp_send_err(
            req,
            HTTPD_500_INTERNAL_SERVER_ERROR,
            "No camera frame available yet");
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(
        req,
        "Cache-Control",
        "no-cache, no-store, must-revalidate");

    // Keep the mutex while sending so the published buffer cannot
    // be modified while the HTTP server is reading it.
    esp_err_t ret = httpd_resp_send(
        req,
        (const char *)g_latest_jpeg,
        g_latest_jpeg_len);

    xSemaphoreGive(g_publish_mutex);

    return ret;
}

// ============================================================
// HTTP: Depth map
// ============================================================

static esp_err_t http_depth_handler(httpd_req_t *req)
{
    if (g_depth_data == nullptr || g_publish_mutex == nullptr)
    {
        httpd_resp_send_err(
            req,
            HTTPD_500_INTERNAL_SERVER_ERROR,
            "Depth buffer unavailable");
        return ESP_FAIL;
    }

    if (xSemaphoreTake(g_publish_mutex, pdMS_TO_TICKS(2000)) != pdTRUE)
    {
        httpd_resp_send_err(
            req,
            HTTPD_500_INTERNAL_SERVER_ERROR,
            "Depth temporarily busy");
        return ESP_FAIL;
    }

    if (!g_frame_ready)
    {
        xSemaphoreGive(g_publish_mutex);

        httpd_resp_send_err(
            req,
            HTTPD_500_INTERNAL_SERVER_ERROR,
            "No depth frame available yet");
        return ESP_FAIL;
    }

    static uint8_t packet_header[16];

    packet_header[0] = 'D';
    packet_header[1] = 'E';
    packet_header[2] = 'P';
    packet_header[3] = '1';

    packet_header[4] = (uint8_t)(DEPTH_WIDTH & 0xFF);
    packet_header[5] = (uint8_t)((DEPTH_WIDTH >> 8) & 0xFF);
    packet_header[6] = (uint8_t)(DEPTH_HEIGHT & 0xFF);
    packet_header[7] = (uint8_t)((DEPTH_HEIGHT >> 8) & 0xFF);

    packet_header[8] = (uint8_t)(int8_t)g_depth_exponent;

    packet_header[9] = 0;
    packet_header[10] = 0;
    packet_header[11] = 0;

    packet_header[12] = (uint8_t)(DEPTH_SIZE & 0xFF);
    packet_header[13] = (uint8_t)((DEPTH_SIZE >> 8) & 0xFF);
    packet_header[14] = (uint8_t)((DEPTH_SIZE >> 16) & 0xFF);
    packet_header[15] = (uint8_t)((DEPTH_SIZE >> 24) & 0xFF);

    httpd_resp_set_type(req, "application/octet-stream");
    httpd_resp_set_hdr(
        req,
        "Cache-Control",
        "no-cache, no-store, must-revalidate");
    httpd_resp_set_hdr(
        req,
        "Access-Control-Allow-Origin",
        "*");

    esp_err_t ret = httpd_resp_send_chunk(
        req,
        (const char *)packet_header,
        sizeof(packet_header));

    if (ret != ESP_OK)
    {
        xSemaphoreGive(g_publish_mutex);
        return ret;
    }

    const size_t CHUNK_SIZE = 2048;
    size_t sent = 0;

    while (sent < DEPTH_SIZE)
    {
        size_t remaining = DEPTH_SIZE - sent;
        size_t chunk = remaining > CHUNK_SIZE ? CHUNK_SIZE : remaining;

        ret = httpd_resp_send_chunk(
            req,
            (const char *)(g_depth_data + sent),
            chunk);

        if (ret != ESP_OK)
        {
            httpd_resp_send_chunk(req, nullptr, 0);
            xSemaphoreGive(g_publish_mutex);
            return ret;
        }

        sent += chunk;
    }

    ret = httpd_resp_send_chunk(req, nullptr, 0);

    xSemaphoreGive(g_publish_mutex);

    return ret;
}

// ============================================================
// HTTP server
// ============================================================

static httpd_handle_t start_webserver()
{
    httpd_config_t config =
        HTTPD_DEFAULT_CONFIG();

    config.server_port = 80;

    // Increase URI handler/task stack slightly because
    // the camera/depth endpoints are active together.
    config.stack_size = 8192;

    httpd_handle_t server = nullptr;

    if (httpd_start(
            &server,
            &config
        ) == ESP_OK)
    {
        httpd_uri_t root_uri = {};

        root_uri.uri = "/";
        root_uri.method = HTTP_GET;
        root_uri.handler = http_root_handler;
        root_uri.user_ctx = nullptr;

        ESP_ERROR_CHECK(
            httpd_register_uri_handler(
                server,
                &root_uri
            )
        );

        httpd_uri_t camera_uri = {};

        camera_uri.uri = "/camera";
        camera_uri.method = HTTP_GET;
        camera_uri.handler = http_camera_handler;
        camera_uri.user_ctx = nullptr;

        ESP_ERROR_CHECK(
            httpd_register_uri_handler(
                server,
                &camera_uri
            )
        );

        httpd_uri_t depth_uri = {};

        depth_uri.uri = "/depth";
        depth_uri.method = HTTP_GET;
        depth_uri.handler = http_depth_handler;
        depth_uri.user_ctx = nullptr;

        ESP_ERROR_CHECK(
            httpd_register_uri_handler(
                server,
                &depth_uri
            )
        );

        ESP_LOGI(TAG, "HTTP server started.");

        return server;
    }

    ESP_LOGE(TAG, "Failed to start HTTP server.");

    return nullptr;
}

// ============================================================
// Resize RGB888 camera image to 128x96
// ============================================================

static void resize_rgb888_to_model(
    const uint8_t *src,
    int src_w,
    int src_h,
    float *dst)
{
    // Center crop to 4:3 first, then resize.
    // QVGA is already 4:3, so this is a direct resize.
    for (int y = 0; y < DEPTH_HEIGHT; y++)
    {
        int sy =
            (y * src_h) / DEPTH_HEIGHT;

        if (sy >= src_h)
            sy = src_h - 1;

        for (int x = 0; x < DEPTH_WIDTH; x++)
        {
            int sx =
                (x * src_w) / DEPTH_WIDTH;

            if (sx >= src_w)
                sx = src_w - 1;

            int src_idx =
                (sy * src_w + sx) * 3;

            int dst_idx =
                (y * DEPTH_WIDTH + x) * 3;

            dst[dst_idx + 0] =
                src[src_idx + 0] / 255.0f;

            dst[dst_idx + 1] =
                src[src_idx + 1] / 255.0f;

            dst[dst_idx + 2] =
                src[src_idx + 2] / 255.0f;
        }
    }
}

// ============================================================
// Run inference on one live camera frame
// ============================================================

static bool run_camera_inference(dl::Model *model)
{
    if (g_inference_busy)
        return false;

    g_inference_busy = true;

    bool success = false;

    camera_fb_t *fb = esp_camera_fb_get();

    if (fb == nullptr)
    {
        ESP_LOGE(TAG, "Camera capture failed for inference.");
        g_inference_busy = false;
        return false;
    }

    // Copy the JPEG from this exact camera frame before returning the
    // camera frame buffer. It will be published together with the
    // resulting depth map after inference completes.
    if (fb->format != PIXFORMAT_JPEG || fb->len > MAX_JPEG_SIZE)
    {
        ESP_LOGE(TAG,
                 "Unexpected camera frame: format=%d len=%u",
                 (int)fb->format,
                 (unsigned)fb->len);
        esp_camera_fb_return(fb);
        g_inference_busy = false;
        return false;
    }

    if (g_pending_jpeg == nullptr)
    {
        g_pending_jpeg =
            (uint8_t *)heap_caps_malloc(
                MAX_JPEG_SIZE,
                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

        if (g_pending_jpeg == nullptr)
        {
            ESP_LOGE(TAG, "Failed to allocate pending JPEG buffer.");
            esp_camera_fb_return(fb);
            g_inference_busy = false;
            return false;
        }
    }

    memcpy(g_pending_jpeg, fb->buf, fb->len);
    g_pending_jpeg_len = fb->len;

    // Decode the SAME captured frame to RGB888 for the neural network.
    size_t rgb_size =
        (size_t)CAMERA_WIDTH *
        (size_t)CAMERA_HEIGHT *
        3;

    if (g_camera_rgb == nullptr)
    {
        g_camera_rgb =
            (uint8_t *)heap_caps_malloc(
                rgb_size,
                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

        if (g_camera_rgb == nullptr)
        {
            ESP_LOGE(TAG, "Failed to allocate RGB camera buffer.");
            esp_camera_fb_return(fb);
            g_inference_busy = false;
            return false;
        }
    }

    bool converted =
        fmt2rgb888(
            fb->buf,
            fb->len,
            fb->format,
            g_camera_rgb);

    esp_camera_fb_return(fb);

    if (!converted)
    {
        ESP_LOGE(TAG, "JPEG -> RGB888 conversion failed.");
        g_inference_busy = false;
        return false;
    }

    auto inputs = model->get_inputs();
    auto outputs = model->get_outputs();

    if (inputs.empty() || outputs.empty())
    {
        ESP_LOGE(TAG, "Model input/output unavailable.");
        g_inference_busy = false;
        return false;
    }

    dl::TensorBase *model_input = inputs.begin()->second;
    dl::TensorBase *model_output = outputs.begin()->second;

    if (model_input->get_size() != MODEL_RGB_SIZE)
    {
        ESP_LOGE(TAG, "Unexpected input size: %d",
                 model_input->get_size());
        g_inference_busy = false;
        return false;
    }

    if (model_output->get_size() != DEPTH_SIZE)
    {
        ESP_LOGE(TAG, "Unexpected output size: %d",
                 model_output->get_size());
        g_inference_busy = false;
        return false;
    }

    if (g_model_input_float == nullptr)
    {
        g_model_input_float =
            (float *)heap_caps_malloc(
                MODEL_RGB_SIZE * sizeof(float),
                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

        if (g_model_input_float == nullptr)
        {
            ESP_LOGE(TAG, "Failed to allocate model input buffer.");
            g_inference_busy = false;
            return false;
        }
    }

    resize_rgb888_to_model(
        g_camera_rgb,
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        g_model_input_float);

    dl::TensorBase float_input(
        {1, DEPTH_HEIGHT, DEPTH_WIDTH, 3},
        g_model_input_float,
        0,
        dl::DATA_TYPE_FLOAT,
        false);

    if (!model_input->assign(&float_input))
    {
        ESP_LOGE(TAG, "Failed to assign RGB camera input.");
        g_inference_busy = false;
        return false;
    }

    int64_t start_us = esp_timer_get_time();

    model->run(dl::RUNTIME_MODE_SINGLE_CORE);

    int64_t end_us = esp_timer_get_time();

    g_inference_ms =
        (float)(end_us - start_us) / 1000.0f;

    if (g_inference_ms > 0.0f)
        g_fps = 1000.0f / g_inference_ms;
    else
        g_fps = 0.0f;

    int8_t *output_data =
        model_output->get_element_ptr<int8_t>();

    if (output_data == nullptr)
    {
        ESP_LOGE(TAG, "Output pointer is NULL.");
        g_inference_busy = false;
        return false;
    }

    g_depth_exponent = (int)model_output->exponent;

    const float output_scale =
        DL_SCALE(model_output->exponent);

    // Prepare the new depth map first.
    memcpy(g_depth_data, output_data, DEPTH_SIZE);

    float min_depth = FLT_MAX;
    float max_depth = -FLT_MAX;
    double sum_depth = 0.0;

    for (int i = 0; i < DEPTH_SIZE; i++)
    {
        float depth =
            dl::dequantize(
                g_depth_data[i],
                output_scale);

        if (depth < min_depth)
            min_depth = depth;

        if (depth > max_depth)
            max_depth = depth;

        sum_depth += depth;
    }

    float mean_depth =
        (float)(sum_depth / DEPTH_SIZE);

    // ------------------------------------------------------------
    // Atomic publication:
    // JPEG and depth become visible to the browser together.
    // Until this point, the browser keeps seeing the previous pair.
    // ------------------------------------------------------------
    if (g_publish_mutex != nullptr &&
        xSemaphoreTake(g_publish_mutex, pdMS_TO_TICKS(2000)) == pdTRUE)
    {
        if (g_latest_jpeg != nullptr &&
            g_pending_jpeg != nullptr &&
            g_pending_jpeg_len <= MAX_JPEG_SIZE)
        {
            memcpy(
                g_latest_jpeg,
                g_pending_jpeg,
                g_pending_jpeg_len);

            g_latest_jpeg_len = g_pending_jpeg_len;

            g_min_depth = min_depth;
            g_max_depth = max_depth;
            g_mean_depth = mean_depth;

            g_frame_ready = true;
        }

        xSemaphoreGive(g_publish_mutex);
        success = g_frame_ready;
    }
    else
    {
        ESP_LOGE(TAG, "Failed to lock frame publication mutex.");
    }

    ESP_LOGI(
        TAG,
        "LIVE frame published: %.3f ms | %.2f FPS | depth %.4f..%.4f mean %.4f | JPEG %u bytes",
        g_inference_ms,
        g_fps,
        min_depth,
        max_depth,
        mean_depth,
        (unsigned)g_pending_jpeg_len);

    g_inference_busy = false;

    return success;
}

// ============================================================
// Background inference task
// ============================================================

static void inference_task(void *arg)
{
    dl::Model *model =
        (dl::Model *)arg;

    ESP_LOGI(TAG,
             "Live camera inference task started.");

    while (true)
    {
        if (!g_inference_busy)
        {
            run_camera_inference(model);
        }

        // The camera + model pipeline is intentionally
        // sequential to avoid overlapping inference.
        vTaskDelay(
            pdMS_TO_TICKS(50)
        );
    }
}

// ============================================================
// APP MAIN
// ============================================================

extern "C" void app_main(void)
{
    ESP_LOGI(TAG,
             "==============================================");

    ESP_LOGI(TAG,
             "Depth Anything V2 V1 Student");

    ESP_LOGI(TAG,
             "ESP32-S3 / ESP-DL / Live RGB Camera");

    ESP_LOGI(TAG,
             "Input: live RGB camera -> 128x96 RGB");

    ESP_LOGI(TAG,
             "Output: 128x96 INT8 depth");

    ESP_LOGI(TAG,
             "==============================================");

    print_memory(
        "Memory before initialization:"
    );

    // --------------------------------------------------------
    // Camera
    // --------------------------------------------------------

    if (!camera_init())
    {
        ESP_LOGE(TAG,
                 "Camera initialization failed.");

        return;
    }

    // --------------------------------------------------------
    // Wi-Fi
    // --------------------------------------------------------

    wifi_init_ap();

    // --------------------------------------------------------
    // Model
    // --------------------------------------------------------

    ESP_LOGI(TAG,
             "Loading ESP-DL model...");

    dl::Model *model =
        new dl::Model(
            (const char *)model_espdl,
            fbs::MODEL_LOCATION_IN_FLASH_RODATA,
            0,
            dl::MEMORY_MANAGER_GREEDY,
            nullptr,
            true
        );

    if (model == nullptr)
    {
        ESP_LOGE(TAG,
                 "Model allocation failed!");

        return;
    }

    ESP_LOGI(TAG,
             "Model loaded successfully.");

    print_memory(
        "Memory after model loading:"
    );

    // --------------------------------------------------------
    // ESP-PPQ model test
    // --------------------------------------------------------

    ESP_LOGI(TAG,
             "Running embedded ESP-PPQ model test...");

    esp_err_t test_result =
        model->test();

    if (test_result == ESP_OK)
    {
        ESP_LOGI(TAG,
                 "ESP-DL MODEL TEST: PASSED");
    }
    else
    {
        ESP_LOGE(TAG,
                 "ESP-DL MODEL TEST: FAILED");

        delete model;

        return;
    }

    // --------------------------------------------------------
    // Allocate persistent depth buffer.
    // --------------------------------------------------------

    g_depth_data =
        (int8_t *)heap_caps_malloc(
            DEPTH_SIZE,
            MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
        );

    if (g_depth_data == nullptr)
    {
        ESP_LOGE(TAG,
                 "Failed to allocate depth buffer.");

        delete model;

        return;
    }

    memset(
        g_depth_data,
        0,
        DEPTH_SIZE
    );

    // --------------------------------------------------------
    // Shared published-frame buffers and mutex.
    // --------------------------------------------------------

    g_publish_mutex = xSemaphoreCreateMutex();

    if (g_publish_mutex == nullptr)
    {
        ESP_LOGE(TAG, "Failed to create frame publication mutex.");
        delete model;
        return;
    }

    g_pending_jpeg =
        (uint8_t *)heap_caps_malloc(
            MAX_JPEG_SIZE,
            MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    g_latest_jpeg =
        (uint8_t *)heap_caps_malloc(
            MAX_JPEG_SIZE,
            MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    if (g_pending_jpeg == nullptr ||
        g_latest_jpeg == nullptr)
    {
        ESP_LOGE(TAG, "Failed to allocate JPEG publication buffers.");
        delete model;
        return;
    }

    memset(g_pending_jpeg, 0, MAX_JPEG_SIZE);
    memset(g_latest_jpeg, 0, MAX_JPEG_SIZE);

    // --------------------------------------------------------
    // Start live inference task.
    // --------------------------------------------------------

    BaseType_t task_ret =
        xTaskCreatePinnedToCore(
            inference_task,
            "depth_inference",
            8192,
            model,
            5,
            nullptr,
            1
        );

    if (task_ret != pdPASS)
    {
        ESP_LOGE(TAG,
                 "Failed to start inference task.");

        delete model;

        return;
    }

    // --------------------------------------------------------
    // Start HTTP server.
    // --------------------------------------------------------

    if (start_webserver() == nullptr)
    {
        ESP_LOGE(TAG,
                 "HTTP server failed to start.");

        delete model;

        return;
    }

    print_memory(
        "Memory after startup:"
    );

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG,
             "==============================================");

    ESP_LOGI(TAG,
             "LIVE DEPTH STREAM READY");

    ESP_LOGI(TAG,
             "Wi-Fi SSID: DEPTH_V1_ESP32S3");

    ESP_LOGI(TAG,
             "Password : depth123");

    ESP_LOGI(TAG,
             "Browser  : http://192.168.4.1");

    ESP_LOGI(TAG,
             "Camera   : /camera");

    ESP_LOGI(TAG,
             "Depth    : /depth");

    ESP_LOGI(TAG,
             "==============================================");

    while (true)
    {
        vTaskDelay(
            pdMS_TO_TICKS(5000)
        );
    }
}
