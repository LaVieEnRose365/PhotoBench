# PhotoBench Dataset Overview

This folder provides two components for evaluation and research:
- Samples: a small, display-oriented subset with images for demonstration only.
- Protected: machine-readable features without raw images for local evaluation.

## Directory Layout

```
dataset/
├── README.md            # English overview (this file)
├── README_CN.md         # Chinese overview
├── samples/             # Display subset (per album, 20 queries with display images)
│   ├── album1/
│   │   ├── query.json   # 20 queries with GroundTruth image filenames
│   │   └── images/      # display images referenced by GroundTruth
│   ├── album2/
│   └── album3/
│   ├── README.md
│   └── sample_queries.schema.json
└── protected/           # Features for local evaluation (no raw images)
    ├── album1/
    │   ├── query.json   # all queries (no GroundTruth fields)
    │   └── images/
    │       ├── captions/    # text indices by model
    │       ├── embeddings/  # vector indices by model
    │       ├── metadata/
    │       │   └── metadata.json  # location/time metadata (privacy-preserving)
    │       └── faceid/
    │           ├── face_info_cn.json
    │           ├── face_info_en.json
    │           └── face_id_images/
    ├── album2/
    └── album3/
```

## Files and Field Definitions

### samples/query.json (20 per album)
- Type: list of query objects.
- Key fields:
  - `Query`: text of the retrieval intent.
  - `Query_id`: query identifier.
  - `Album_id`: album identifier.
  - `Source_type`: source category label.
  - `Reasoning_type`: reasoning category (e.g., location/time/social).
  - `GroundTruth`: list of image filenames used for display/visualization.
- Purpose: visualization for readers and reviewers; not for training or full scoring.

### protected/query.json (album-level, all queries)
- Type: list of query objects.
- Contains the same query fields as the samples version except any GroundTruth-related fields are omitted.

### protected/images/captions/
- Per-model text indices for caption-based retrieval.
- Each `metadata.json` includes:
  - `model_name`: the name of the caption model.
  - `filenames`: list of image filenames included in the index.

### protected/images/embeddings/
- Per-model vector indices for image embedding retrieval.
- Each `metadata.json` includes:
  - `model_name`
  - `filenames`

### protected/images/metadata/metadata.json
- Privacy-preserving location and time metadata for images.
- Fields:
  - `latitude`: decimal degrees rounded to approximately 10 meters.
  - `longitude`: decimal degrees rounded to approximately 10 meters.
  - `time`: ISO-format timestamp string.

### protected/images/faceid/
- Anonymized person information to support social/identity reasoning without exposing raw identities.
- Files:
  - `face_info_cn.json` / `face_info_en.json` include:
    - `face_id_to_nicknames`: mapping from face IDs to anonymized names/nicknames.
    - `image_to_face_ids`: mapping from image filename to a list of face IDs present.
  - `face_id_images/`: representative images for face IDs that appear.

## Intended Use and Evaluation
- Samples: for display and qualitative understanding; not for model training.
- Protected: for local algorithms to run against machine-readable features.
- Evaluation: submit results (e.g., `submission.json`) for scoring against hidden GroundTruth.

For privacy and compliance notes, refer to the project’s instruction documents.
