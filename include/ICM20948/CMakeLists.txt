aux_source_directory(. DIR_icm20948_SRCS)

add_library(ICM20948 ${DIR_icm20948_SRCS})
target_include_directories(ICM20948
PUBLIC
  .
)
target_link_libraries(ICM20948 PUBLIC hardware_i2c pico_stdlib)
