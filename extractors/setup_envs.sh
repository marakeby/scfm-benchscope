#!/bin/bash
# Setup per-model virtual environments
#
# Usage: ./setup_envs.sh [model]
#   ./setup_envs.sh          # Setup all environments
#   ./setup_envs.sh scgpt    # Setup only scgpt environment
#   ./setup_envs.sh stack    # Setup only stack environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Python version to use (defaults to python3, can override with PYTHON_VERSION env var)
PYTHON_VERSION=${PYTHON_VERSION:-python3}

setup_stack() {
    echo "=== Setting up STACK environment ==="
    mkdir -p stack
    cd stack
    
    # Set up log file
    LOG_FILE="install.log"
    echo "Installation log will be saved to: $SCRIPT_DIR/stack/$LOG_FILE"
    echo "=== STACK Environment Setup - $(date) ===" | tee "$LOG_FILE"
    
    # Check if Python version exists
    if ! command -v $PYTHON_VERSION &> /dev/null; then
        echo "Error: $PYTHON_VERSION not found. Please install Python 3.10+ or set PYTHON_VERSION env var" | tee -a "$LOG_FILE"
        cd ..
        return 1
    fi
    
    if [ ! -d "env" ]; then
        echo "Creating virtual environment with $PYTHON_VERSION..." | tee -a "$LOG_FILE"
        $PYTHON_VERSION -m venv env 2>&1 | tee -a "$LOG_FILE"
        if [ ! -f "env/bin/activate" ]; then
            echo "Error: Failed to create virtual environment" | tee -a "$LOG_FILE"
            cd ..
            return 1
        fi
    else
        echo "Environment already exists, skipping creation" | tee -a "$LOG_FILE"
    fi
    
    source env/bin/activate
    
    # Use the pip from this environment explicitly
    env/bin/pip install --upgrade pip 2>&1 | tee -a "$LOG_FILE"
    
    # Install torch matching main environment (2.7.1+cu128) before installing STACK
    # This ensures CUDA compatibility and consistency
    echo "Installing PyTorch 2.7.1+cu128 to match main environment..." | tee -a "$LOG_FILE"
    env/bin/pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 2>&1 | tee -a "$LOG_FILE"
    
    # STACK uses pyproject.toml with setuptools
    # Install from local repo - it will pull in dependencies from pyproject.toml
    echo "Installing STACK from local repo..." | tee -a "$LOG_FILE"
    env/bin/pip install -e /lotterlab/users/riccardo/ML_BIO/Bio_FMs/RNA/stack 2>&1 | tee -a "$LOG_FILE"
    
    deactivate
    cd ..
    echo "✓ STACK environment ready at: $SCRIPT_DIR/stack/env" | tee -a "$SCRIPT_DIR/stack/$LOG_FILE"
    echo "✓ Installation log saved to: $SCRIPT_DIR/stack/$LOG_FILE"
}

setup_scgpt() {
    echo "=== Setting up scGPT environment ==="
    mkdir -p scgpt
    cd scgpt
    
    # Set up log file
    LOG_FILE="install.log"
    echo "Installation log will be saved to: $SCRIPT_DIR/scgpt/$LOG_FILE"
    echo "=== scGPT Environment Setup - $(date) ===" | tee "$LOG_FILE"
    
    if [ ! -d "env" ]; then
        echo "Creating virtual environment with $PYTHON_VERSION..." | tee -a "$LOG_FILE"
        $PYTHON_VERSION -m venv env 2>&1 | tee -a "$LOG_FILE"
        if [ ! -f "env/bin/activate" ]; then
            echo "Error: Failed to create virtual environment" | tee -a "$LOG_FILE"
            cd ..
            return 1
        fi
    else
        echo "Environment already exists, skipping creation" | tee -a "$LOG_FILE"
    fi
    
    source env/bin/activate
    
    # Use the pip from this environment explicitly
    env/bin/pip install --upgrade pip 2>&1 | tee -a "$LOG_FILE"
    
    # scGPT requires torch 2.0.1 specifically for torchtext compatibility
    # Cannot use 2.7.1 because torchtext 0.15.2 is incompatible with newer torch
    echo "Installing PyTorch 2.0.1 for scGPT (required for torchtext compatibility)..." | tee -a "$LOG_FILE"
    env/bin/pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 2>&1 | tee -a "$LOG_FILE"
    env/bin/pip install torchtext==0.15.2 2>&1 | tee -a "$LOG_FILE"
    
    # scGPT uses Poetry (pyproject.toml with poetry-core)
    # Modern pip can handle Poetry projects, but we need poetry-core
    echo "Installing poetry-core for scGPT..." | tee -a "$LOG_FILE"
    env/bin/pip install poetry-core 2>&1 | tee -a "$LOG_FILE"
    
    # Install scGPT from local repo - pip will use poetry-core to read dependencies
    echo "Installing scGPT from local repo..." | tee -a "$LOG_FILE"
    env/bin/pip install -e /lotterlab/users/riccardo/ML_BIO/Bio_FMs/RNA/scGPT 2>&1 | tee -a "$LOG_FILE"
    
    deactivate
    cd ..
    echo "✓ scGPT environment ready at: $SCRIPT_DIR/scgpt/env" | tee -a "$SCRIPT_DIR/scgpt/$LOG_FILE"
    echo "✓ Installation log saved to: $SCRIPT_DIR/scgpt/$LOG_FILE"
}

setup_geneformer() {
    echo "=== Setting up Geneformer environment ==="
    mkdir -p geneformer
    cd geneformer
    
    # Set up log file
    LOG_FILE="install.log"
    echo "Installation log will be saved to: $SCRIPT_DIR/geneformer/$LOG_FILE"
    echo "=== Geneformer Environment Setup - $(date) ===" | tee "$LOG_FILE"
    
    if [ ! -d "env" ]; then
        echo "Creating virtual environment with $PYTHON_VERSION..." | tee -a "$LOG_FILE"
        $PYTHON_VERSION -m venv env 2>&1 | tee -a "$LOG_FILE"
        if [ ! -f "env/bin/activate" ]; then
            echo "Error: Failed to create virtual environment" | tee -a "$LOG_FILE"
            cd ..
            return 1
        fi
    else
        echo "Environment already exists, skipping creation" | tee -a "$LOG_FILE"
    fi
    
    source env/bin/activate
    
    # Use the pip from this environment explicitly
    env/bin/pip install --upgrade pip 2>&1 | tee -a "$LOG_FILE"
    
    # Install torch matching main environment (2.7.1+cu128) before other deps
    echo "Installing PyTorch 2.7.1+cu128 to match main environment..." | tee -a "$LOG_FILE"
    env/bin/pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 2>&1 | tee -a "$LOG_FILE"
    
    # Install extractor requirements (transformers, datasets, scanpy, etc.) INTO this env
    echo "Installing Geneformer extractor dependencies from requirements.txt..." | tee -a "$LOG_FILE"
    env/bin/pip install -r "$SCRIPT_DIR/geneformer/requirements.txt" 2>&1 | tee -a "$LOG_FILE"
    
    # Geneformer package (tokenizer, token_dictionary, etc.)
    echo "Installing Geneformer from local repo..." | tee -a "$LOG_FILE"
    env/bin/pip install -e /lotterlab/users/riccardo/ML_BIO/Bio_FMs/RNA/Geneformer 2>&1 | tee -a "$LOG_FILE"
    
    deactivate
    cd ..
    echo "✓ Geneformer environment ready at: $SCRIPT_DIR/geneformer/env" | tee -a "$SCRIPT_DIR/geneformer/$LOG_FILE"
    echo "  The extractor calls this env when run via run_exp (needs_separate_env)." | tee -a "$SCRIPT_DIR/geneformer/$LOG_FILE"
    echo "✓ Installation log saved to: $SCRIPT_DIR/geneformer/$LOG_FILE"
}

setup_state() {
    echo "=== Setting up STATE environment ==="
    mkdir -p state
    cd state
    
    # Set up log file
    LOG_FILE="install.log"
    echo "Installation log will be saved to: $SCRIPT_DIR/state/$LOG_FILE"
    echo "=== STATE Environment Setup - $(date) ===" | tee "$LOG_FILE"
    
    if [ ! -d "env" ]; then
        echo "Creating virtual environment with $PYTHON_VERSION..." | tee -a "$LOG_FILE"
        $PYTHON_VERSION -m venv env 2>&1 | tee -a "$LOG_FILE"
        if [ ! -f "env/bin/activate" ]; then
            echo "Error: Failed to create virtual environment" | tee -a "$LOG_FILE"
            cd ..
            return 1
        fi
    else
        echo "Environment already exists, skipping creation" | tee -a "$LOG_FILE"
    fi
    
    source env/bin/activate
    
    # Use the pip from this environment explicitly
    env/bin/pip install --upgrade pip 2>&1 | tee -a "$LOG_FILE"
    
    # STATE requires torch>=2.7.0, so we can use the main env version
    echo "Installing PyTorch 2.7.1+cu128 to match main environment..." | tee -a "$LOG_FILE"
    env/bin/pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 2>&1 | tee -a "$LOG_FILE"
    
    # STATE uses pyproject.toml with hatchling
    # Install from local repo - it will pull in dependencies from pyproject.toml
    echo "Installing STATE from local repo..." | tee -a "$LOG_FILE"
    env/bin/pip install -e /lotterlab/users/riccardo/ML_BIO/Bio_FMs/RNA/state 2>&1 | tee -a "$LOG_FILE"
    
    deactivate
    cd ..
    echo "✓ STATE environment ready at: $SCRIPT_DIR/state/env" | tee -a "$SCRIPT_DIR/state/$LOG_FILE"
    echo "✓ Installation log saved to: $SCRIPT_DIR/state/$LOG_FILE"
}

setup_uce() {
    echo "=== Setting up UCE environment ==="
    mkdir -p uce
    cd uce
    
    # Set up log file
    LOG_FILE="install.log"
    echo "Installation log will be saved to: $SCRIPT_DIR/uce/$LOG_FILE"
    echo "=== UCE Environment Setup - $(date) ===" | tee "$LOG_FILE"
    
    if [ ! -d "env" ]; then
        echo "Creating virtual environment with $PYTHON_VERSION..." | tee -a "$LOG_FILE"
        $PYTHON_VERSION -m venv env 2>&1 | tee -a "$LOG_FILE"
        if [ ! -f "env/bin/activate" ]; then
            echo "Error: Failed to create virtual environment" | tee -a "$LOG_FILE"
            cd ..
            return 1
        fi
    else
        echo "Environment already exists, skipping creation" | tee -a "$LOG_FILE"
    fi
    
    source env/bin/activate
    
    # Use the pip from this environment explicitly
    env/bin/pip install --upgrade pip 2>&1 | tee -a "$LOG_FILE"
    
    # Install torch matching main environment (2.7.1+cu128) before installing UCE deps
    echo "Installing PyTorch 2.7.1+cu128 to match main environment..." | tee -a "$LOG_FILE"
    env/bin/pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 2>&1 | tee -a "$LOG_FILE"
    
    # UCE is not a proper package (no setup.py/pyproject.toml)
    # Install dependencies from requirements.txt
    echo "Installing UCE dependencies from requirements.txt..." | tee -a "$LOG_FILE"
    env/bin/pip install -r /lotterlab/users/riccardo/ML_BIO/Bio_FMs/RNA/UCE/requirements.txt 2>&1 | tee -a "$LOG_FILE"
    
    # UCE code will be added to PYTHONPATH in the extract script
    
    deactivate
    cd ..
    echo "✓ UCE environment ready at: $SCRIPT_DIR/uce/env" | tee -a "$SCRIPT_DIR/uce/$LOG_FILE"
    echo "  Note: UCE code is added to PYTHONPATH at runtime" | tee -a "$SCRIPT_DIR/uce/$LOG_FILE"
    echo "✓ Installation log saved to: $SCRIPT_DIR/uce/$LOG_FILE"
}

setup_aido() {
    echo "=== Setting up AIDO.Cell environment ==="
    mkdir -p aido
    cd aido
    
    # Set up log file
    LOG_FILE="install.log"
    echo "Installation log will be saved to: $SCRIPT_DIR/aido/$LOG_FILE"
    echo "=== AIDO.Cell Environment Setup - $(date) ===" | tee "$LOG_FILE"
    
    if [ ! -d "env" ]; then
        echo "Creating virtual environment with $PYTHON_VERSION..." | tee -a "$LOG_FILE"
        $PYTHON_VERSION -m venv env 2>&1 | tee -a "$LOG_FILE"
        if [ ! -f "env/bin/activate" ]; then
            echo "Error: Failed to create virtual environment" | tee -a "$LOG_FILE"
            cd ..
            return 1
        fi
    else
        echo "Environment already exists, skipping creation" | tee -a "$LOG_FILE"
    fi
    
    source env/bin/activate
    
    # Use the pip from this environment explicitly
    env/bin/pip install --upgrade pip 2>&1 | tee -a "$LOG_FILE"
    
    # Install torch matching main environment (2.7.1+cu128)
    echo "Installing PyTorch 2.7.1+cu128 to match main environment..." | tee -a "$LOG_FILE"
    env/bin/pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 2>&1 | tee -a "$LOG_FILE"
    
    # AIDO.Cell uses HuggingFace transformers
    # Need transformers 5.x (pre-release) for huggingface-hub>=1.0 compatibility
    echo "Installing AIDO.Cell dependencies..." | tee -a "$LOG_FILE"
    env/bin/pip install --pre transformers>=5.0.0 2>&1 | tee -a "$LOG_FILE"
    env/bin/pip install scanpy anndata numpy scipy pandas tqdm safetensors 2>&1 | tee -a "$LOG_FILE"
    
    # Note: AIDO.Cell model is loaded from HuggingFace Hub, not a local repo
    
    deactivate
    cd ..
    echo "✓ AIDO.Cell environment ready at: $SCRIPT_DIR/aido/env" | tee -a "$SCRIPT_DIR/aido/$LOG_FILE"
    echo "✓ Installation log saved to: $SCRIPT_DIR/aido/$LOG_FILE"
}

setup_scfoundation() {
    echo "=== Setting up scFoundation environment ==="
    mkdir -p scfoundation
    cd scfoundation
    
    # Set up log file
    LOG_FILE="install.log"
    echo "Installation log will be saved to: $SCRIPT_DIR/scfoundation/$LOG_FILE"
    echo "=== scFoundation Environment Setup - $(date) ===" | tee "$LOG_FILE"
    
    if [ ! -d "env" ]; then
        echo "Creating virtual environment with $PYTHON_VERSION..." | tee -a "$LOG_FILE"
        $PYTHON_VERSION -m venv env 2>&1 | tee -a "$LOG_FILE"
        if [ ! -f "env/bin/activate" ]; then
            echo "Error: Failed to create virtual environment" | tee -a "$LOG_FILE"
            cd ..
            return 1
        fi
    else
        echo "Environment already exists, skipping creation" | tee -a "$LOG_FILE"
    fi
    
    source env/bin/activate
    
    # Use the pip from this environment explicitly
    env/bin/pip install --upgrade pip 2>&1 | tee -a "$LOG_FILE"
    
    # Install torch matching main environment (2.7.1+cu128) before installing scFoundation deps
    echo "Installing PyTorch 2.7.1+cu128 to match main environment..." | tee -a "$LOG_FILE"
    env/bin/pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 2>&1 | tee -a "$LOG_FILE"
    
    # scFoundation code is in the eval repo's features/scfoundation directory
    # Install dependencies from requirements.txt
    echo "Installing scFoundation dependencies from requirements.txt..." | tee -a "$LOG_FILE"
    env/bin/pip install -r "$SCRIPT_DIR/scfoundation/requirements.txt" 2>&1 | tee -a "$LOG_FILE"
    
    # Note: scFoundation code is in scFM_eval/features/scfoundation and will be added to PYTHONPATH at runtime
    
    deactivate
    cd ..
    echo "✓ scFoundation environment ready at: $SCRIPT_DIR/scfoundation/env" | tee -a "$SCRIPT_DIR/scfoundation/$LOG_FILE"
    echo "  Note: scFoundation code is in features/scfoundation and added to PYTHONPATH at runtime" | tee -a "$SCRIPT_DIR/scfoundation/$LOG_FILE"
    echo "✓ Installation log saved to: $SCRIPT_DIR/scfoundation/$LOG_FILE"
}

# Main logic
if [ $# -eq 0 ]; then
    echo "Setting up all environments..."
    setup_stack
    setup_scgpt
    setup_geneformer
    setup_state
    setup_uce
    setup_aido
    setup_scfoundation
    echo ""
    echo "=== All environments created ==="
else
    case "$1" in
        stack)
            setup_stack
            ;;
        scgpt)
            setup_scgpt
            ;;
        geneformer)
            setup_geneformer
            ;;
        state)
            setup_state
            ;;
        uce)
            setup_uce
            ;;
        aido)
            setup_aido
            ;;
        scfoundation)
            setup_scfoundation
            ;;
        *)
            echo "Unknown model: $1"
            echo "Available: stack, scgpt, geneformer, state, uce, aido, scfoundation"
            exit 1
            ;;
    esac
fi

echo ""
echo "To use an environment:"
echo "  source extractors/<model>/env/bin/activate"
echo "  python extractors/<model>/extract.py --help"
