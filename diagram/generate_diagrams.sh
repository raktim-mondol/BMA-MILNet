#!/bin/bash

# Generate all BMA MIL Classifier diagrams

echo "Generating BMA MIL Classifier diagrams..."

# System Architecture
echo "1. Generating System Architecture..."
mmdc -i system_architecture.mmd -o system_architecture.png -s 3

# Component Architecture
echo "2. Generating Component Architecture..."
mmdc -i component_architecture.mmd -o component_architecture.png -s 3

# Workflow
echo "3. Generating Workflow..."
mmdc -i workflow.mmd -o workflow.png -s 3

# Data Flow
echo "4. Generating Data Flow..."
mmdc -i data_flow.mmd -o data_flow.png -s 3

# Training Pipeline
echo "5. Generating Training Pipeline..."
mmdc -i training_pipeline.mmd -o training_pipeline.png -s 3

# Model Architecture
echo "6. Generating Model Architecture..."
mmdc -i model_architecture.mmd -o model_architecture.png -s 3

echo "All diagrams generated successfully!"
echo "Generated files:"
ls -la *.png