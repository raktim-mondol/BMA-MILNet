# BMA MIL Classifier - Architecture & Workflow Diagrams

## 1. System Architecture Diagram

```mermaid
graph TB
    %% Input Layer
    subgraph "Input: Pile of Images"
        A1[Image 1<br/>4032×3024]
        A2[Image 2<br/>4032×3024]
        A3[Image N<br/>4032×3024]
    end

    %% Patch Extraction Layer
    subgraph "Patch Extraction"
        B1[Patch Extractor]
        B2[12 Patches per Image<br/>1008×1008]
        B3[Resize to 224×224]
    end

    %% Feature Extraction Layer
    subgraph "Feature Extraction"
        C1[ViT-R50 Model<br/>vit_base_r50_s16_224.orig_in21k]
        C2[768-dimensional<br/>Features per Patch]
    end

    %% Image Level Aggregation
    subgraph "Image Level Aggregation"
        D1[Patch Features<br/>12×768]
        D2[Attention Mechanism]
        D3[Weighted Sum]
        D4[Image Representation<br/>512-dim]
    end

    %% Pile Level Aggregation
    subgraph "Pile Level Aggregation"
        E1[Image Features<br/>N×512]
        E2[Attention Mechanism]
        E3[Weighted Sum]
        E4[Pile Representation<br/>256-dim]
    end

    %% Classification Layer
    subgraph "Classification"
        F1[Classifier Network]
        F2[BMA Class 1-4<br/>Prediction]
    end

    %% Data Flow
    A1 --> B1
    A2 --> B1
    A3 --> B1

    B1 --> B2
    B2 --> B3
    B3 --> C1

    C1 --> C2
    C2 --> D1

    D1 --> D2
    D2 --> D3
    D3 --> D4

    D4 --> E1
    E1 --> E2
    E2 --> E3
    E3 --> E4

    E4 --> F1
    F1 --> F2

    %% Styling
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef patch fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef feature fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef imageAgg fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef pileAgg fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef output fill:#e8eaf6,stroke:#283593,stroke-width:2px

    class A1,A2,A3 input
    class B1,B2,B3 patch
    class C1,C2 feature
    class D1,D2,D3,D4 imageAgg
    class E1,E2,E3,E4 pileAgg
    class F1,F2 output
```

## 2. Detailed Component Architecture

```mermaid
flowchart TD
    %% Main Components
    subgraph "BMA MIL Classifier System"
        PE[PatchExtractor<br/>4032×3024 → 12×224×224]
        FE[FeatureExtractor<br/>ViT-R50 Pre-trained]
        ILA[ImageLevelAggregator<br/>Attention over 12 patches]
        PLA[PileLevelAggregator<br/>Attention over N images]
        CLS[Classifier<br/>4-class BMA]
    end

    %% Data Flow
    IMG[4032×3024 Image] --> PE
    PE --> PATCHES[12×224×224 Patches]
    PATCHES --> FE
    FE --> FEATURES[12×768 Features]
    FEATURES --> ILA
    ILA --> IMG_FEAT[512-dim Image Feature]
    IMG_FEAT --> PLA
    PLA --> PILE_FEAT[256-dim Pile Feature]
    PILE_FEAT --> CLS
    CLS --> PRED[BMA Prediction 1-4]

    %% Attention Mechanisms
    subgraph "Attention Details"
        PATCH_ATT[Patch Attention<br/>Linear→Tanh→Linear→Softmax]
        IMG_ATT[Image Attention<br/>Linear→Tanh→Linear→Softmax]
    end

    ILA -.-> PATCH_ATT
    PLA -.-> IMG_ATT

    %% Styling
    classDef component fill:#bbdefb,stroke:#1976d2,stroke-width:2px
    classDef data fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    classDef attention fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef output fill:#f8bbd9,stroke:#c2185b,stroke-width:2px

    class PE,FE,ILA,PLA,CLS component
    class IMG,PATCHES,FEATURES,IMG_FEAT,PILE_FEAT data
    class PATCH_ATT,IMG_ATT attention
    class PRED output
```

## 3. Overall Workflow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Dataset
    participant PatchExtractor
    participant FeatureExtractor
    participant ImageAggregator
    participant PileAggregator
    participant Classifier
    participant Results

    Note over User,Dataset: Initialization Phase
    User->>Dataset: Load BWM_label_data.csv
    Dataset-->>User: 123 piles, 4 BMA classes

    Note over User,FeatureExtractor: Feature Extraction Phase
    loop For each pile
        loop For each image in pile
            User->>PatchExtractor: Process 4032×3024 image
            PatchExtractor->>PatchExtractor: Extract 12×1008×1008 patches
            PatchExtractor->>PatchExtractor: Resize to 12×224×224
            PatchExtractor->>FeatureExtractor: Send patches
            FeatureExtractor->>FeatureExtractor: ViT-R50 forward pass
            FeatureExtractor-->>ImageAggregator: 12×768 features
        end
    end

    Note over ImageAggregator,PileAggregator: Aggregation Phase
    loop For each pile
        ImageAggregator->>ImageAggregator: Compute patch attention weights
        ImageAggregator->>ImageAggregator: Weighted sum → 512-dim image features
        ImageAggregator->>PileAggregator: Send image features
        PileAggregator->>PileAggregator: Compute image attention weights
        PileAggregator->>PileAggregator: Weighted sum → 256-dim pile feature
        PileAggregator->>Classifier: Send pile feature
    end

    Note over Classifier,Results: Classification Phase
    Classifier->>Classifier: Forward pass through MLP
    Classifier-->>Results: 4-class probabilities
    Results-->>User: Final BMA prediction

    Note over User,Results: Evaluation
    User->>Results: Calculate accuracy, F1 score
    Results-->>User: Performance metrics
```

## 4. Data Flow & Dimensions Diagram

```mermaid
graph LR
    %% Input Dimensions
    A[4032×3024×3 Image] --> B[12 Patches]
    B --> C[1008×1008×3 per Patch]
    C --> D[Resize to 224×224×3]
    D --> E[ViT-R50 Input]

    %% Feature Dimensions
    E --> F[768-dim Features<br/>per Patch]
    F --> G[12×768 Features<br/>per Image]
    G --> H[Image Aggregation]
    H --> I[512-dim Image Feature]

    %% Pile Dimensions
    I --> J[N×512 Features<br/>per Pile]
    J --> K[Pile Aggregation]
    K --> L[256-dim Pile Feature]
    L --> M[4-class BMA Prediction]

    %% Styling
    classDef inputDims fill:#e3f2fd,stroke:#1565c0
    classDef patchDims fill:#f3e5f5,stroke:#7b1fa2
    classDef featureDims fill:#e8f5e8,stroke:#2e7d32
    classDef aggregation fill:#fff3e0,stroke:#ef6c00
    classDef outputDims fill:#fce4ec,stroke:#c2185b

    class A,B,C,D inputDims
    class E,F,G patchDims
    class H,I featureDims
    class J,K,L aggregation
    class M outputDims
```

## 5. Training Pipeline Diagram

```mermaid
flowchart TB
    subgraph "Training Pipeline"
        A[Load Dataset<br/>BWM_label_data.csv]
        B[Split Piles<br/>70% Train, 15% Val, 15% Test]
        C[Initialize Model<br/>BMA_MIL_Classifier]
        D[Setup Optimizer<br/>Adam, lr=1e-4]
        E[Training Loop]
        F[Validation]<br/>
        G[Save Best Model]
        H[Test Evaluation]
    end

    subgraph "Training Loop Details"
        E1[For each epoch]
        E2[For each batch of piles]
        E3[Extract patches & features]
        E4[Forward pass through model]
        E5[Compute loss<br/>CrossEntropyLoss]
        E6[Backward pass & update]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> E1
    E1 --> E2
    E2 --> E3
    E3 --> E4
    E4 --> E5
    E5 --> E6
    E6 --> F
    F --> G
    G --> H

    %% Styling
    classDef pipeline fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    classDef loop fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef eval fill:#e8f5e8,stroke:#388e3c,stroke-width:2px

    class A,B,C,D,E,F,G,H pipeline
    class E1,E2,E3,E4,E5,E6 loop
```

## 6. Model Architecture Details

```mermaid
classDiagram
    class PatchExtractor {
        -patch_size: int
        -target_size: int
        +extract_patches(image_path)
        -divide_into_patches(image)
        -resize_patches(patches)
    }

    class FeatureExtractor {
        -model: ViT-R50
        -transform: transforms
        -device: str
        +extract_features(patches)
        -load_model()
        -setup_transforms()
    }

    class ImageLevelAggregator {
        -attention: Sequential
        -image_encoder: Sequential
        +forward(patch_features)
        -compute_attention_weights(features)
        -weighted_sum(features, weights)
    }

    class PileLevelAggregator {
        -attention: Sequential
        -classifier: Sequential
        +forward(image_features)
        -compute_attention_weights(features)
        -weighted_sum(features, weights)
    }

    class BMA_MIL_Classifier {
        -image_aggregator: ImageLevelAggregator
        -pile_aggregator: PileLevelAggregator
        +forward(patch_features_list)
        -aggregate_patches_to_images(patch_features)
        -aggregate_images_to_pile(image_features)
    }

    class BMADataset {
        -data_df: DataFrame
        -image_dir: str
        -feature_extractor: FeatureExtractor
        -pile_groups: dict
        +__getitem__(idx)
        -extract_pile_features(pile_name)
    }

    PatchExtractor --> FeatureExtractor : provides patches
    FeatureExtractor --> BMADataset : extracts features
    BMADataset --> BMA_MIL_Classifier : provides data
    BMA_MIL_Classifier --> ImageLevelAggregator : uses for image aggregation
    BMA_MIL_Classifier --> PileLevelAggregator : uses for pile aggregation
```

## Key Features Illustrated:

1. **Multi-level Hierarchy**: Patch → Image → Pile aggregation
2. **Attention Mechanisms**: Both patch-level and image-level attention
3. **Variable Input Support**: Handles different numbers of images per pile
4. **Pre-trained Features**: ViT-R50 for robust feature extraction
5. **End-to-end Pipeline**: Complete workflow from raw images to BMA predictions

These diagrams provide comprehensive visualization of both the architectural design and operational workflow of the BMA MIL classifier system.