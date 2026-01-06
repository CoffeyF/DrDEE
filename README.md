# DRDEE: Core Implementation

This repository contains the core implementation code for DRDEE (Document-level Relation-aware Document-level Event Extraction), focusing on triple construction validation and record reorganization for the ChFinAnn dataset.

## Code Structure

```
drdee/
├── model.py          # Core model architecture
├── train.py          # Training procedures
├── postprocess.py    # Post-processing and conflict resolution
├── evaluate.py       # Evaluation metrics
├── utils.py          # Utility functions
└── schema.py         # Event schema definitions
```

## Dataset

This implementation is validated on the **ChFinAnn** dataset.

## Related Work

- **HDICL**: Prompt templates and instructions for HDICL are provided in the paper appendix.
- **NER Components**: NER-related code and implementations will be released upon paper acceptance.
- **Duffin Dataset**: Code and data processing scripts for the Duffin dataset will be made available after paper acceptance.

## Note

This repository contains only the core implementation code for DRDEE. Complete code including:
- HDICL and NER implementations
- Duffin dataset processing scripts
- Full data preprocessing pipelines

will be released upon paper acceptance in a separate repository.
