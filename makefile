CXX=g++-5
CXXFLAGS=-c -std=c++14
LDFLAGS=-lopencv_core -lopencv_highgui
OBJECTS=main.o
EXECUTABLE=contrast

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(EXECUTABLE) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS)
