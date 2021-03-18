if (NOT TARGET arducam)
    add_library(arducam INTERFACE)

    pico_generate_pio_header(arducam
      ${CMAKE_CURRENT_LIST_DIR}/image.pio)

    target_sources(arducam INTERFACE
      ${CMAKE_CURRENT_LIST_DIR}/arducam.c)

    target_link_libraries(arducam INTERFACE pico_stdlib hardware_i2c hardware_pio hardware_dma)
endif()
   