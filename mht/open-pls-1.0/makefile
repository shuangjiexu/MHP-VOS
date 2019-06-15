
BUILD_DIR = build
SRC_DIR   = src
BIN_DIR   = bin

CFLAGS = -Winline -O2 -std=c++0x -g
#CFLAGS = -Winline -DDEBUG_MESSAGE -O0 -std=c++0x -g

SOURCES_TMP += Isolates4.cpp
SOURCES_TMP += GraphTools.cpp
SOURCES_TMP += PhasedLocalSearch.cpp
SOURCES_TMP += CliquePhasedLocalSearch.cpp
SOURCES_TMP += IndependentSetPhasedLocalSearch.cpp
SOURCES_TMP += Tools.cpp

SOURCES=$(addprefix $(SOURCES_DIR)/, $(SOURCES_TMP))

OBJECTS_TMP=$(SOURCES_TMP:.cpp=.o)
OBJECTS=$(addprefix $(BUILD_DIR)/, $(OBJECTS_TMP))

DEPFILES_TMP:=$(SOURCES_TMP:.cpp=.d)
DEPFILES=$(addprefix $(BUILD_DIR)/, $(DEPFILES_TMP))

EXEC_NAMES = pls

EXECS = $(addprefix $(BIN_DIR)/, $(EXEC_NAMES))

VPATH = src

.PHONY : all

all: $(EXECS)

.PHONY : clean

clean: 
	rm -rf $(EXECS) $(BUILD_DIR) $(BIN_DIR)

$(BIN_DIR)/pls: main.cpp ${OBJECTS} | ${BIN_DIR}
	g++ $(CFLAGS) -D GIT_COMMIT=`git log --format="%H" -n 1` -D GIT_STATUS="`git status -s -uno`" ${DEFINE} ${OBJECTS} $(SRC_DIR)/main.cpp -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(SRC_DIR)/%.h $(BUILD_DIR)/%.d | $(BUILD_DIR)
	g++ $(CFLAGS) ${DEFINE} -c $< -o $@

$(BUILD_DIR)/%.d: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	g++ $(CFLAGS) -MM -MT '$(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$<)' $< -MF $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

