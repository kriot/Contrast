CXX=g++-5
CXXFLAGS=-g -c -std=c++14
LDFLAGS=-lopencv_core -lopencv_highgui -lopencv_imgproc
OBJECTS=kernel.o
EXECUTABLE=contrast

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(EXECUTABLE) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS)
