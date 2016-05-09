STD := -lstdc++fs
BOOST := -lboost_iostreams -lboost_serialization 
OPENCV := -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn -lopencv_dpm -lopencv_fuzzy -lopencv_line_descriptor -lopencv_optflow -lopencv_plot -lopencv_reg -lopencv_saliency -lopencv_structured_light -lopencv_surface_matching -lopencv_text -lopencv_face -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ximgproc -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core

OBJ := ./obj
BIN := ./bin

all: train_compile main_compile

exp: $(BIN) exp.cpp
	g++ exp.cpp -o $(BIN)/exp.out
	$(BIN)/exp.out

# start executables
train: train_compile
	$(BIN)/train.app
	
main: main_compile
	$(BIN)/main.app

# executables
train_compile: $(BIN) classificator.o train.o
	g++ $(OBJ)/classificator.o $(OBJ)/train.o $(STD) $(BOOST) $(OPENCV) -o $(BIN)/train.app
	
main_compile: $(BIN) classificator.o main.o
	g++ $(OBJ)/classificator.o $(OBJ)/main.o $(STD) $(BOOST) $(OPENCV) -o $(BIN)/main.app

# objects
train.o: $(OBJ) train.cpp
	g++ -c train.cpp -o $(OBJ)/train.o

main.o: $(OBJ) main.cpp
	g++ -c main.cpp -o $(OBJ)/main.o

classificator.o: $(OBJ) classificator.cpp
	g++ -c classificator.cpp -o $(OBJ)/classificator.o

$(OBJ):
	mkdir -p $(OBJ)

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -r -f $(OBJ) $(BIN)

make_dirs:
	mkdir -p $(OBJ) $(BIN)
