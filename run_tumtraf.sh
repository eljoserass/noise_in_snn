#!/bin/bash
# Legacy TUMTraf runner. Use run_dsec.sh for active DSEC workflows.

# Usage: ./run_tumtraf.sh [STAGE] [OPTIONS]
# Stages: preprocess, train-ann, train-snn, eval, pipeline, pipeline-both
# Examples:
#   ./run_tumtraf.sh preprocess --all --rewrite
#   ./run_tumtraf.sh train-ann --rgb --epochs 50
#   ./run_tumtraf.sh train-snn --eb --epochs 50
#   ./run_tumtraf.sh eval --model-path checkpoints/model.pth
#   ./run_tumtraf.sh pipeline --rgb  # runs full pipeline: preprocess -> train -> eval
#   ./run_tumtraf.sh pipeline --eb   # runs full pipeline for event-based data
#   ./run_tumtraf.sh pipeline-both --parallel  # runs both RGB and EB pipelines in parallel
#   ./run_tumtraf.sh pipeline-both  # runs both RGB and EB pipelines sequentially

show_usage() {
    echo "Usage: ./run_tumtraf.sh [STAGE] [OPTIONS]"
    echo ""
    echo "Stages:"
    echo "  preprocess       - Preprocess data (supports --rgb, --eb, --all)"
    echo "  train-ann        - Train ANN on RGB data"
    echo "  train-snn        - Train SNN on event-based data"
    echo "  eval             - Evaluate trained model"
    echo "  pipeline         - Run full pipeline (preprocess -> train -> eval)"
    echo "                     Use --rgb or --eb to specify data type"
    echo "  pipeline-both    - Run both RGB and EB pipelines"
    echo "                     Use --parallel to run them in parallel (requires 2 GPUs for optimal performance)"
    echo ""
    echo "Examples:"
    echo "  ./run_tumtraf.sh preprocess --all --rewrite"
    echo "  ./run_tumtraf.sh train-ann --epochs 50 --lr 0.001"
    echo "  ./run_tumtraf.sh pipeline --rgb"
    echo "  ./run_tumtraf.sh pipeline-both                    # Sequential (1 GPU needed)"
    echo "  ./run_tumtraf.sh pipeline-both --parallel         # Parallel (2 GPUs optimal"
    echo ""
}

# Check for help flag
if [ $# -eq 0 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_usage
    exit 0
fi

# maybe should check the python version we can work with

# check .venv is created, if not create it and install requirements
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

STAGE=$1
shift  # Remove first argument (stage) so remaining args can be passed to scripts

# Detect available GPUs
detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
        echo $NUM_GPUS
    else
        echo 0
    fi
}

NUM_GPUS=$(detect_gpus)

# Route to appropriate script based on stage
case $STAGE in
    preprocess)
        echo "Running preprocessing..."
        python scripts/preprocess.py "$@"
        ;;
    train-ann)
        echo "Running ANN training..."
        python scripts/train_ann_rgb.py "$@"
        ;;
    train-snn)
        echo "Running SNN training..."
        python scripts/train_snn_eb.py "$@"
        ;;
    eval)
        echo "Running evaluation..."
        # Note: Requires --model-path and --model-type arguments
        python scripts/evaluate.py "$@"
        ;;
    pipeline)
        echo "Running full pipeline..."
        
        # Preprocess
        echo "Step 1/3: Preprocessing..."
        python scripts/preprocess.py "$@"
        if [ $? -ne 0 ]; then
            echo "Preprocessing failed!"
            exit 1
        fi
        
        # Determine which training script to run based on arguments
        if [[ "$*" == *"--eb"* ]]; then
            echo "Step 2/3: Training SNN (event-based)..."
            python scripts/train_snn_eb.py "$@"
        else
            echo "Step 2/3: Training ANN (RGB)..."
            python scripts/train_ann_rgb.py "$@"
        fi
        
        if [ $? -ne 0 ]; then
            echo "Training failed!"
            exit 1
        fi
        
        # Evaluate
        echo "Step 3/3: Evaluating..."
        # Determine model path from training output (assumes checkpoint saved to checkpoints/)
        if [[ "$*" == *"--eb"* ]]; then
            MODEL_TYPE="snn"
            CHECKPOINT="checkpoints/vgg11_ssd_snn_best.pth"
        else
            MODEL_TYPE="ann"
            CHECKPOINT="checkpoints/vgg11_ssd_ann_best.pth"
        fi
        
        if [ ! -f "$CHECKPOINT" ]; then
            echo "Warning: Checkpoint not found at $CHECKPOINT"
            echo "Skipping evaluation. Run evaluation manually with:"
            echo "  python scripts/evaluate.py --model-path <path> --model-type $MODEL_TYPE"
        else
            python scripts/evaluate.py --model-path "$CHECKPOINT" --model-type "$MODEL_TYPE" "$@"
            if [ $? -ne 0 ]; then
                echo "Evaluation failed!"
                exit 1
            fi
        fi
        
        echo "Pipeline completed successfully!"
        ;;
    pipeline-both)
        echo "Running both RGB and EB pipelines..."
        
        # Check if parallel flag is present
        PARALLEL=false
        FILTERED_ARGS=""
        for arg in "$@"; do
            if [ "$arg" == "--parallel" ]; then
                PARALLEL=true
            else
                FILTERED_ARGS="$FILTERED_ARGS $arg"
            fi
        done
        
        if [ "$PARALLEL" = true ]; then
            echo "Running pipelines in PARALLEL mode..."
            
            # Check GPU availability
            if [ $NUM_GPUS -eq 0 ]; then
                echo "ERROR: No CUDA GPUs detected. Cannot run parallel training."
                echo "Please check nvidia-smi or run in sequential mode."
                exit 1
            elif [ $NUM_GPUS -eq 1 ]; then
                echo "WARNING: Only 1 GPU detected. Parallel mode will run both models on the same GPU."
                echo "This is typically SLOWER than sequential mode due to resource contention."
                echo "Recommendation: Use sequential mode (remove --parallel flag) for better performance."
                read -p "Continue anyway? [y/N] " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    echo "Aborted. Run without --parallel for sequential mode."
                    exit 0
                fi
                GPU_ANN=0
                GPU_SNN=0
                echo "Proceeding with both models on GPU 0..."
            else
                echo "✓ Detected $NUM_GPUS GPUs. Assigning GPU 0 → ANN, GPU 1 → SNN"
                GPU_ANN=0
                GPU_SNN=1
            fi
            
            # Preprocess both (must be sequential as it's the same script)
            echo "Step 1/3: Preprocessing both RGB and EB..."
            python scripts/preprocess.py --all $FILTERED_ARGS
            if [ $? -ne 0 ]; then
                echo "Preprocessing failed!"
                exit 1
            fi
            
            # Train both in parallel with GPU assignment
            echo "Step 2/3: Training ANN (GPU $GPU_ANN) and SNN (GPU $GPU_SNN) in parallel..."
            CUDA_VISIBLE_DEVICES=$GPU_ANN python scripts/train_ann_rgb.py $FILTERED_ARGS &
            PID_ANN=$!
            CUDA_VISIBLE_DEVICES=$GPU_SNN python scripts/train_snn_eb.py $FILTERED_ARGS &
            PID_SNN=$!
            
            # Wait for both to complete
            wait $PID_ANN
            EXIT_ANN=$?
            wait $PID_SNN
            EXIT_SNN=$?
            
            if [ $EXIT_ANN -ne 0 ]; then
                echo "ANN training failed!"
                exit 1
            fi
            if [ $EXIT_SNN -ne 0 ]; then
                echo "SNN training failed!"
                exit 1
            fi
            
            # Evaluate both in parallel with GPU assignment
            echo "Step 3/3: Evaluating both models in parallel..."
            if [ -f "checkpoints/vgg11_ssd_ann_best.pth" ]; then
                CUDA_VISIBLE_DEVICES=$GPU_ANN python scripts/evaluate.py --model-path "checkpoints/vgg11_ssd_ann_best.pth" --model-type ann $FILTERED_ARGS &
                PID_EVAL_ANN=$!
            else
                echo "Warning: ANN checkpoint not found, skipping ANN evaluation"
                PID_EVAL_ANN=0
            fi
            
            if [ -f "checkpoints/vgg11_ssd_snn_best.pth" ]; then
                CUDA_VISIBLE_DEVICES=$GPU_SNN python scripts/evaluate.py --model-path "checkpoints/vgg11_ssd_snn_best.pth" --model-type snn $FILTERED_ARGS &
                PID_EVAL_SNN=$!
            else
                echo "Warning: SNN checkpoint not found, skipping SNN evaluation"
                PID_EVAL_SNN=0
            fi
            
            if [ $PID_EVAL_ANN -ne 0 ]; then
                wait $PID_EVAL_ANN
                EXIT_EVAL_ANN=$?
                if [ $EXIT_EVAL_ANN -ne 0 ]; then
                    echo "ANN evaluation failed!"
                    exit 1
                fi
            fi
            
            if [ $PID_EVAL_SNN -ne 0 ]; then
                wait $PID_EVAL_SNN
                EXIT_EVAL_SNN=$?
                if [ $EXIT_EVAL_SNN -ne 0 ]; then
                    echo "SNN evaluation failed!"
                    exit 1
                fi
            fi
            
            echo "Both pipelines completed successfully!"
        else
            echo "Running pipelines in SEQUENTIAL mode..."
            
            # Preprocess both
            echo "Step 1/6: Preprocessing both RGB and EB..."
            python scripts/preprocess.py --all $FILTERED_ARGS
            if [ $? -ne 0 ]; then
                echo "Preprocessing failed!"
                exit 1
            fi
            
            # RGB Pipeline
            echo "Step 2/6: Training ANN (RGB)..."
            python scripts/train_ann_rgb.py $FILTERED_ARGS
            if [ $? -ne 0 ]; then
                echo "ANN training failed!"
                exit 1
            fi
            
            echo "Step 3/6: Evaluating ANN (RGB)..."
            if [ -f "checkpoints/vgg11_ssd_ann_best.pth" ]; then
                python scripts/evaluate.py --model-path "checkpoints/vgg11_ssd_ann_best.pth" --model-type ann $FILTERED_ARGS
                if [ $? -ne 0 ]; then
                    echo "ANN evaluation failed!"
                    exit 1
                fi
            else
                echo "Warning: ANN checkpoint not found, skipping evaluation"
            fi
            
            # EB Pipeline
            echo "Step 4/6: Training SNN (EB)..."
            python scripts/train_snn_eb.py $FILTERED_ARGS
            if [ $? -ne 0 ]; then
                echo "SNN training failed!"
                exit 1
            fi
            
            echo "Step 5/6: Evaluating SNN (EB)..."
            if [ -f "checkpoints/vgg11_ssd_snn_best.pth" ]; then
                python scripts/evaluate.py --model-path "checkpoints/vgg11_ssd_snn_best.pth" --model-type snn $FILTERED_ARGS
                if [ $? -ne 0 ]; then
                    echo "SNN evaluation failed!"
                    exit 1
                fi
            else
                echo "Warning: SNN checkpoint not found, skipping evaluation"
            fi
            
            echo "Both pipelines completed successfully!"
        fi
        ;;
    -h|--help)
        show_usage
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo ""
        show_usage
        exit 1
        ;;
esac
