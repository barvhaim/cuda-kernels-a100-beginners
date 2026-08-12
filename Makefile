BUILD_DIR ?= build

.PHONY: configure build test run run-foundations run-patterns run-basics run-llm profile clean

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80

build: configure
	cmake --build $(BUILD_DIR) -j

test:
	python3 -m unittest discover -s tests -v

run-foundations: build
	bash scripts/run_all.sh $(BUILD_DIR) foundations

run-patterns: build
	bash scripts/run_all.sh $(BUILD_DIR) patterns

run-basics: build
	bash scripts/run_all.sh $(BUILD_DIR) basics

run-llm: build
	bash scripts/run_all.sh $(BUILD_DIR) llm

run: build
	bash scripts/run_all.sh $(BUILD_DIR) all

profile: build
	ncu --set basic --kernel-name vector_add $(BUILD_DIR)/06_vector_add

clean:
	rm -rf $(BUILD_DIR)
