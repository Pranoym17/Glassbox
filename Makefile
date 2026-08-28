.PHONY: vector-add clean

BUILD_DIR := build
VECTOR_ADD := $(BUILD_DIR)/vector_add

vector-add: $(VECTOR_ADD)
	$(VECTOR_ADD)

$(VECTOR_ADD): cuda/vector_add.cu | $(BUILD_DIR)
	nvcc -O2 -o $@ $<

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
