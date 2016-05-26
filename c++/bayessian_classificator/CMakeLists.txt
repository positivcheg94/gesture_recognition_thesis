cmake_minimum_required(VERSION 3.5)
project(bayessian_classificator)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

set(STD "-lstdc++fs")
set(OPENCVLIBS "-lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn -lopencv_dpm -lopencv_fuzzy -lopencv_line_descriptor -lopencv_optflow -lopencv_plot -lopencv_reg -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_rgbd -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_face -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_ml -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core")
set(BOOSTLIBS "-lboost_iostreams -lboost_serialization")
set(REALSENSE "-lrealsense")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${OPENCVLIBS} ${BOOSTLIBS} ${STD} ${REALSENSE}")

add_library(continuous_matrix continuous_matrix.cpp)
add_library(classificator classificator.cpp)


add_executable(process_opencv process_opencv.cpp)
add_executable(process_realsense process_realsense.cpp)
add_executable(train_from_folder train_from_folder.cpp)
add_executable(train_from_realsense train_from_realsense.cpp)
add_executable(filter_model filter_model.cpp)
add_executable(show_model_representation show_model_representation.cpp)

target_link_libraries(process_opencv continuous_matrix classificator)
target_link_libraries(process_realsense continuous_matrix classificator)
target_link_libraries(train_from_folder continuous_matrix classificator)
target_link_libraries(train_from_realsense continuous_matrix classificator)
target_link_libraries(filter_model continuous_matrix classificator)
target_link_libraries(show_model_representation continuous_matrix classificator)

