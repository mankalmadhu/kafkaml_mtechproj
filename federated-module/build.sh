#!/bin/bash

# Configuration
REGISTRY=${REGISTRY:-"host.docker.internal:50000"}
MODULES=("federated_backend" "federated_model_training/tensorflow" "federated_model_control_logger" "federated_data_control_logger")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print usage
print_usage() {
    echo "Usage: $0 [command] [module]"
    echo ""
    echo "Commands:"
    echo "  build [module]    Build Docker image for specified module"
    echo "  build-all         Build Docker images for all modules"
    echo "  push [module]     Push Docker image for specified module"
    echo "  push-all          Push Docker images for all modules"
    echo ""
    echo "Modules:"
    for module in "${MODULES[@]}"; do
        if [[ $module == *"/"* ]]; then
            echo "  $module"
        else
            echo "  $(basename $module)"
        fi
    done
    echo ""
    echo "Examples:"
    echo "  $0 build federated_model_training/tensorflow"
    echo "  $0 build federated_backend"
    echo "  $0 push federated_model_training/tensorflow"
    echo "  $0 build-all"
    echo "  $0 push-all"
}

# Function to get module path from name
get_module_path() {
    local module_name=$1
    for module in "${MODULES[@]}"; do
        if [ "$module" = "$module_name" ] || [ "$(basename $module)" = "$module_name" ]; then
            # If there are multiple matches with the same basename, require full path
            local matches=0
            local matched_module=""
            for m in "${MODULES[@]}"; do
                if [ "$(basename $m)" = "$(basename $module)" ]; then
                    matches=$((matches + 1))
                    matched_module=$m
                fi
            done
            
            if [ $matches -gt 1 ] && [ "$module_name" = "$(basename $module)" ]; then
                echo -e "${RED}Error: Multiple modules found with name '$module_name'. Please use full path:${NC}"
                for m in "${MODULES[@]}"; do
                    if [ "$(basename $m)" = "$module_name" ]; then
                        echo "  $m"
                    fi
                done
                return 1
            fi
            echo $module
            return 0
        fi
    done
    return 1
}

# Function to get image name from module path
get_image_name() {
    local module_path=$1
    if [[ $module_path == *"/"* ]]; then
        # Replace / with _ for image name
        echo "${module_path//\//_}"
    else
        echo "$module_path"
    fi
}

REGISTRY=${REGISTRY:-"host.docker.internal:50000"}
MODULES=("federated_backend" "federated_model_training/tensorflow" "federated_model_control_logger" "federated_data_control_logger")


# Function to build a single module
build_module() {
    local module_name=$1
    local module_path=$(get_module_path $module_name)
    
    if [ -z "$module_path" ]; then
        echo -e "${RED}Error: Module '$module_name' not found${NC}"
        print_usage
        exit 1
    fi

    local image_name=$(get_image_name $module_path)
    echo -e "${YELLOW}Building $module_path (image: $image_name)...${NC}"
    
    # Special handling for tensorflow module
    if [[ $module_path == "federated_model_training/tensorflow" ]]; then
        docker build -t $REGISTRY/$image_name:latest --build-arg TFTAG=2.7.0 $module_path
    else
        docker build -t $REGISTRY/$image_name:latest $module_path
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully built $module_path${NC}"
    else
        echo -e "${RED}Failed to build $module_path${NC}"
        exit 1
    fi
}

# Function to push a single module
push_module() {
    local module_name=$1
    local module_path=$(get_module_path $module_name)
    
    if [ -z "$module_path" ]; then
        echo -e "${RED}Error: Module '$module_name' not found${NC}"
        print_usage
        exit 1
    fi

    local image_name=$(get_image_name $module_path)
    echo -e "${YELLOW}Pushing $module_path (image: $image_name)...${NC}"
    docker push $REGISTRY/$image_name:latest
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully pushed $module_path${NC}"
    else
        echo -e "${RED}Failed to push $module_path${NC}"
        exit 1
    fi
}

# Main script logic
case "$1" in
    "build")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Module name required${NC}"
            print_usage
            exit 1
        fi
        build_module $2
        ;;
    "build-all")
        for module in "${MODULES[@]}"; do
            build_module $module
        done
        ;;
    "push")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Module name required${NC}"
            print_usage
            exit 1
        fi
        push_module $2
        ;;
    "push-all")
        for module in "${MODULES[@]}"; do
            push_module $module
        done
        ;;
    *)
        print_usage
        exit 1
        ;;
esac 