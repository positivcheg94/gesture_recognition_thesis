cmake_minimum_required(VERSION 3.5)
project(bayessian_classificator)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

set(STD "-lstdc++fs")
set(OPENCVLIBS "-lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn -lopencv_dpm -lopencv_fuzzy -lopencv_line_descriptor -lopencv_optflow -lopencv_plot -lopencv_reg -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_rgbd -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_face -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_ml -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core")
set(BOOSTLIBS "-lboost_iostreams -lboost_serialization")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 ${OPENCVLIBS} ${BOOSTLIBS} ${STD}")

set(SOURCE_FILES
        continuous_matrix.cpp
        classificator.cpp
        process_opencv.cpp
        train_from_folder.cpp
        )


add_executable(main process_opencv.cpp)
add_executable(train train_from_folder.cpp)