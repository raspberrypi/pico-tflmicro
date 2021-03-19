if (NOT TARGET arducam_mic)
    add_library(arducam_mic INTERFACE)

    pico_generate_pio_header(arducam_mic
      ${CMAKE_CURRENT_LIST_DIR}/mic_i2s.pio)

    target_sources(arducam_mic INTERFACE
      ${CMAKE_CURRENT_LIST_DIR}/arducam_mic.c
    )
    target_link_libraries(arducam_mic INTERFACE pico_stdlib hardware_i2c hardware_pio hardware_irq hardware_dma)
endif()
   
