# Host (macOS) build of the virtual CODA board. Usage: make -C firmware/host -j8
TFLM := ../lib/tflm
COMMON := ../common
BUILD := build

INC := -I$(COMMON) -I$(TFLM) -I$(TFLM)/third_party/flatbuffers/include \
       -I$(TFLM)/third_party/gemmlowp -I$(TFLM)/third_party/ruy \
       -I$(TFLM)/third_party/kissfft -I$(TFLM)/third_party/cmsis_nn \
       -I$(TFLM)/third_party/cmsis_nn/Include -I$(TFLM)/third_party/cmsis/CMSIS/Core/Include
DEFS := -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -DCMSIS_NN -DCODA_TFLM_TIME \
        -DNDEBUG -DTF_LITE_STRIP_ERROR_STRINGS
OPT := -O2 -ffp-contract=off -ffunction-sections -fdata-sections
CXXFLAGS := -std=c++17 $(OPT) $(DEFS) $(INC) -fno-rtti -fno-exceptions -w
CFLAGS := -std=c17 $(OPT) $(DEFS) $(INC) -w

SRCS := $(shell find $(TFLM)/tensorflow $(TFLM)/third_party/cmsis_nn/Source -name '*.cc' -o -name '*.c' \
          | grep -v cortex_m_generic/micro_time.cc) \
        $(wildcard $(COMMON)/*.cc) $(wildcard $(COMMON)/*.c) main_host.cc
OBJS := $(patsubst %,$(BUILD)/%.o,$(subst ../,,$(SRCS)))

coda_host: $(OBJS)
	$(CXX) -o $@ $^ -Wl,-dead_strip

$(BUILD)/%.cc.o: ../%.cc
	@mkdir -p $(dir $@); $(CXX) $(CXXFLAGS) -c $< -o $@
$(BUILD)/%.c.o: ../%.c
	@mkdir -p $(dir $@); $(CC) $(CFLAGS) -c $< -o $@
$(BUILD)/main_host.cc.o: main_host.cc
	@mkdir -p $(dir $@); $(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD) coda_host
