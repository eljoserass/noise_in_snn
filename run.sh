#!/bin/bash

# Usage: ./run.sh [STAGE] [OPTIONS]
# Stages: preprocess, train-ann, train-snn, eval, pipeline, pipeline-both
# Examples:
#   ./run.sh preprocess --all --rewrite
#   ./run.sh train-ann --rgb --epochs 50
#   ./run.sh train-snn --eb --epochs 50
#   ./run.sh eval --model-path checkpoints/model.pth
#   ./run.sh pipeline --rgb  # runs full pipeline: preprocess -> train -> eval
#   ./run.sh pipeline --eb   # runs full pipeline for event-based data
#   ./run.sh pipeline-both --parallel  # runs both RGB and EB pipelines in parallel
#   ./run.sh pipeline-both  # runs both RGB and EB pipelines sequentially

show_usage() {
    echo "Usage: ./run.sh [STAGE] [OPTIONS]"
    echo ""
    echo "Stages:"
    echo "  preprocess       - Preprocess data (supports --rgb, --eb, --all)"
    echo "  train-ann        - Train ANN on RGB data"
    echo "  train-snn        - Train SNN on event-based data"
    echo "  eval             - Evaluate trained model"
    echo "  pipeline         - Run full pipeline (preprocess -> train -> eval)"
    echo "                     Use --rgb or --eb to specify data type"
    echo "  pipeline-both    - Run both RGB and EB pipelines"
    echo "                     Use --parallel to run them in parallel"
    echo ""
    echo "Examples:"
    echo "  ./run.sh preprocess --all --rewrite"
    echo "  ./run.sh train-ann --epochs 50 --lr 0.001"
    echo "  ./run.sh pipeline --rgb"
    echo "  ./run.sh pipeline-both --parallel"
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

# Route to appropriate script based on stage
case $STAGE in
    preprocess)
        echo "Running preprocessing..."
        python scripts/preprocess.py "$@"
        ;;
    train-ann)
        echo "Running ANN training..."
        python scripts/train_ann.py "$@"
        ;;
    train-snn)
        echo "Running SNN training..."
        python scripts/train_snn.py "$@"
        ;;
    eval)
        echo "Running evaluation..."
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
            python scripts/train_snn.py "$@"
        else
            echo "Step 2/3: Training ANN (RGB)..."
            python scripts/train_ann.py "$@"
        fi
        
        if [ $? -ne 0 ]; then
            echo "Training failed!"
            exit 1
        fi
        
        # Evaluate
        echo "Step 3/3: Evaluating..."
        python scripts/evaluate.py "$@"
        if [ $? -ne 0 ]; then
            echo "Evaluation failed!"
            exit 1
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
            
            # Preprocess both (must be sequential as it's the same script)
            echo "Step 1/3: Preprocessing both RGB and EB..."
            python scripts/preprocess.py --all $FILTERED_ARGS
            if [ $? -ne 0 ]; then
                echo "Preprocessing failed!"
                exit 1
            fi
            
            # Train both in parallel
            echo "Step 2/3: Training ANN and SNN in parallel..."
            python scripts/train_ann.py --rgb $FILTERED_ARGS &
            PID_ANN=$!
            python scripts/train_snn.py --eb $FILTERED_ARGS &
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
            
            # Evaluate both in parallel
            echo "Step 3/3: Evaluating both models in parallel..."
            python scripts/evaluate.py --rgb $FILTERED_ARGS &
            PID_EVAL_ANN=$!
            python scripts/evaluate.py --eb $FILTERED_ARGS &
            PID_EVAL_SNN=$!
            
            wait $PID_EVAL_ANN
            EXIT_EVAL_ANN=$?
            wait $PID_EVAL_SNN
            EXIT_EVAL_SNN=$?
            
            if [ $EXIT_EVAL_ANN -ne 0 ]; then
                echo "ANN evaluation failed!"
                exit 1
            fi
            if [ $EXIT_EVAL_SNN -ne 0 ]; then
                echo "SNN evaluation failed!"
                exit 1
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
            python scripts/train_ann.py --rgb $FILTERED_ARGS
            if [ $? -ne 0 ]; then
                echo "ANN training failed!"
                exit 1
            fi
            
            echo "Step 3/6: Evaluating ANN (RGB)..."
            python scripts/evaluate.py --rgb $FILTERED_ARGS
            if [ $? -ne 0 ]; then
                echo "ANN evaluation failed!"
                exit 1
            fi
            
            # EB Pipeline
            echo "Step 4/6: Training SNN (EB)..."
            python scripts/train_snn.py --eb $FILTERED_ARGS
            if [ $? -ne 0 ]; then
                echo "SNN training failed!"
                exit 1
            fi
            
            echo "Step 5/6: Evaluating SNN (EB)..."
            python scripts/evaluate.py --eb $FILTERED_ARGS
            if [ $? -ne 0 ]; then
                echo "SNN evaluation failed!"
                exit 1
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