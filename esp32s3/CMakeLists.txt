idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES espressif__esp-dl
    PRIV_REQUIRES
        esp_timer
	esp32-camera
        esp_wifi
        esp_event
        esp_netif
        esp_http_server
        nvs_flash
)

# ESP-DL model embedding
set(ESP_DL_FBS_CMAKE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../managed_components/espressif__esp-dl/fbs_loader/cmake"
)

include(
    "${ESP_DL_FBS_CMAKE_DIR}/utilities.cmake"
)

# Explicitly provide the CMake directory expected by
# target_add_aligned_binary_data()
set(cmake_dir "${ESP_DL_FBS_CMAKE_DIR}")

target_add_aligned_binary_data(
    ${COMPONENT_LIB}
    "${CMAKE_CURRENT_LIST_DIR}/model/student_boundary_v1_128x96_esp32s3.espdl"
    BINARY
)