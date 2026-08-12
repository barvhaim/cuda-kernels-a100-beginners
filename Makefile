BUILD_DIR ?= build

.PHONY: configure build test run profile clean

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80

build: configure
	cmake --build $(BUILD_DIR) -j

test:
	python3 -m unittest discover -s tests -v

run: build
	bash scripts/run_all.sh $(BUILD_DIR)

profile: build
	ncu --set basic --kernel-name vector_add $(BUILD_DIR)/01_vector_add

clean:
	rm -rf $(BUILD_DIR)
