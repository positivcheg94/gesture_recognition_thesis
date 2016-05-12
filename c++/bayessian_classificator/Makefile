	GCC := g++ --std=c++11 -O2

STD := -lstdc++fs
BOOST := -lboost_iostreams -lboost_serialization 
OPENCV := -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn -lopencv_dpm -lopencv_fuzzy -lopencv_line_descriptor -lopencv_optflow -lopencv_plot -lopencv_reg -lopencv_saliency -lopencv_structured_light -lopencv_surface_matching -lopencv_text -lopencv_face -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ximgproc -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core
REALSENSE := -lrealsense

OBJ := ./obj
BIN := ./bin
OUTPUT := ./data

.PHONY: clean

all: $(BIN)/train.app $(BIN)/main.app

exp: $(BIN) exp.cpp
	$(GCC) exp.cpp -o $(BIN)/exp.out
	$(BIN)/exp.out

# start executables
train: $(OUTPUT) $(BIN)/train.app
	echo "#! /bin/bash" > train
	echo "$(BIN)/train.app $(OUTPUT) ../../photos_and_masks/3 trained5" >> train
	chmod a+x train
	
main: $(OUTPUT) $(BIN)/main.app
	echo "#! /bin/bash" > main
	echo "$(BIN)/main.app $(OUTPUT) trained5 \$$1" >> main
	chmod a+x main
	

# executables
$(BIN)/train.app: $(BIN) $(OBJ)/continuous_matrix.o $(OBJ)/classificator.o $(OBJ)/train.o
	$(GCC) $(OBJ)/continuous_matrix.o $(OBJ)/classificator.o $(OBJ)/train.o $(STD) $(BOOST) $(OPENCV) -o $@
	
$(BIN)/main.app: $(BIN) $(OBJ)/continuous_matrix.o $(OBJ)/classificator.o $(OBJ)/main.o
	$(GCC) $(OBJ)/continuous_matrix.o $(OBJ)/classificator.o $(OBJ)/main.o $(STD) $(BOOST) $(OPENCV) -o $@

# objects
$(OBJ)/train.o: $(OBJ) train.cpp
	$(GCC) -c train.cpp -o $@

$(OBJ)/main.o: $(OBJ) main.cpp
	$(GCC) -c main.cpp -o $@

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
	rm -r -f $(OBJ) $(BIN) main train
