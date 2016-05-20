	GCC := g++ --std=c++11 -O3

STD := -lstdc++fs
BOOST := -lboost_iostreams -lboost_serialization
OPENCV := -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn -lopencv_dpm -lopencv_fuzzy -lopencv_line_descriptor -lopencv_optflow -lopencv_plot -lopencv_reg -lopencv_saliency -lopencv_structured_light -lopencv_surface_matching -lopencv_text -lopencv_face -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ximgproc -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core
REALSENSE := -lrealsense

OBJ := ./obj
BIN := ./bin
OUTPUT := ./data



TRAIN_FOLDER = ../../photos_and_masks/3
FILENAME := trained6.clr


.PHONY: clean



all: train_from_folder train_from_realsense process_opencv process_realsense



exp: $(BIN) exp.cpp
	$(GCC) exp.cpp -o $(BIN)/exp.out
	$(BIN)/exp.out



# start executables
train_from_folder: $(OUTPUT) $(BIN)/train_from_folder.app
	echo "#! /bin/bash" > train_from_folder
	echo "$(BIN)/train_from_folder.app $(OUTPUT) $(TRAIN_FOLDER) $(FILENAME)" >> train_from_folder
	chmod a+x train_from_folder

train_from_realsense: $(OUTPUT) $(BIN)/train_from_realsense.app
	echo "#! /bin/bash" > train_from_realsense
	echo "$(BIN)/train_from_realsense.app $(FILENAME)" >> train_from_realsense

process_opencv: $(OUTPUT) $(BIN)/process_opencv.app
	echo "#! /bin/bash" > process_opencv
	echo "$(BIN)/process_opencv.app $(OUTPUT) $(FILENAME) \$$1 \$$2" >> process_opencv
	chmod a+x process_opencv

process_realsense: $(OUTPUT) $(BIN)/process_realsense.app
	echo "#! /bin/bash" > process_realsense
	echo "$(BIN)/process_realsense.app $(OUTPUT) $(FILENAME) \$$1 \$$2" >> process_realsense
	chmod a+x process_realsense



# executables
$(BIN)/train_from_folder.app: $(BIN) $(OBJ)/continuous_matrix.o $(OBJ)/classificator.o $(OBJ)/train_from_folder.o
	$(GCC) $(OBJ)/continuous_matrix.o $(OBJ)/classificator.o $(OBJ)/train_from_folder.o $(STD) $(BOOST) $(OPENCV) -o $@

$(BIN)/train_from_realsense.app: $(BIN) $(OBJ)/continuous_matrix.o $(OBJ)/classificator.o $(OBJ)/train_from_realsense.o
	$(GCC) $(OBJ)/continuous_matrix.o $(OBJ)/classificator.o $(OBJ)/train_from_realsense.o $(STD) $(BOOST) $(OPENCV) $(REALSENSE) -o $@

$(BIN)/process_opencv.app: $(BIN) $(OBJ)/continuous_matrix.o $(OBJ)/classificator.o $(OBJ)/process_opencv.o
	$(GCC) $(OBJ)/continuous_matrix.o $(OBJ)/classificator.o $(OBJ)/process_opencv.o $(STD) $(BOOST) $(OPENCV) -o $@

$(BIN)/process_realsense.app: $(BIN) $(OBJ)/continuous_matrix.o $(OBJ)/classificator.o $(OBJ)/process_realsense.o
	$(GCC) $(OBJ)/continuous_matrix.o $(OBJ)/classificator.o $(OBJ)/process_realsense.o $(STD) $(BOOST) $(OPENCV) $(REALSENSE)  -o $@



# objects
$(OBJ)/train_from_realsense.o: train_from_realsense.cpp
	$(GCC) -c train_from_realsense.cpp -o $@

$(OBJ)/train_from_folder.o: $(OBJ) train_from_folder.cpp
	$(GCC) -c train_from_folder.cpp -o $@

$(OBJ)/process_opencv.o: $(OBJ) process_opencv.cpp
	$(GCC) -c process_opencv.cpp -o $@

$(OBJ)/process_realsense.o: $(OBJ) process_realsense.cpp
	$(GCC) -c process_realsense.cpp -o $@

$(OBJ)/classificator.o: $(OBJ) classificator.cpp
	$(GCC) -c classificator.cpp -o $@

$(OBJ)/continuous_matrix.o: $(OBJ) continuous_matrix.cpp
	$(GCC) -c continuous_matrix.cpp -o $@

$(OBJ):
	mkdir -p $(OBJ)

$(BIN):
	mkdir -p $(BIN)

$(OUTPUT):
	mkdir -p $(OUTPUT)

clean:
	rm -r -f $(OBJ) $(BIN) train_from_folder train_from_realsense process_opencv process_realsense
