# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a statistics study repository focused on preparing for the 統計検定準１級 (Statistics Level Pre-1 exam). The project contains Marimo notebooks with educational content about statistical concepts including stochastic processes, Brownian motion, Poisson processes, and regression analysis.

## Development Environment

This project uses:
- **Python 3.11+** as specified in pyproject.toml
- **uv** for dependency management (uv.lock file present)
- **Marimo** for interactive notebooks (main framework for content)

### Key Dependencies
- marimo (interactive Python notebooks)
- numpy, matplotlib, pandas (data analysis and visualization)
- scipy, statsmodels (statistical computations)
- scikit-learn (machine learning utilities)
- japanize-matplotlib (Japanese font support for plots)

## Common Commands

### Environment Setup
```bash
# Install dependencies using uv
uv install

# Or using pip with requirements.txt
pip install -r requirements.txt
```

### Running the Application
```bash
# Run the main entry point
python main.py

# Run specific Marimo notebooks
marimo run src/ch15.py    # Brownian motion and stochastic processes
marimo run src/ch17.py    # Regression diagnostics
marimo run src/ch22.py    # Principal component analysis
```

### Code Quality
```bash
# Format code (ruff is available as dependency)
ruff format .

# Lint code
ruff check .
```

## Code Architecture

### File Structure
- `main.py` - Simple entry point with hello world message
- `src/` - Contains Marimo notebook files with statistical content
  - `ch15.py` - Brownian motion, stochastic processes, Poisson processes
  - `ch17.py` - Linear regression diagnostics, leverage, Cook's distance
  - `ch22.py` - Principal component analysis, SVD, eigendecomposition

### Marimo Notebooks
The educational content is primarily delivered through Marimo interactive notebooks. Each file contains:
- Mathematical explanations in Japanese using `mo.md()` for markdown cells
- Python code demonstrations with numpy, matplotlib, scipy
- Statistical simulations and visualizations
- Educational examples with detailed mathematical derivations

### Content Focus Areas
1. **Stochastic Processes** (ch15.py):
   - Brownian motion theory and properties
   - Poisson processes and applications
   - Relationship with exponential and gamma distributions

2. **Regression Analysis** (ch17.py):
   - Linear regression diagnostics
   - Leverage and influence measures
   - Cook's distance and outlier detection

3. **Multivariate Analysis** (ch22.py):
   - Principal component analysis
   - Singular value decomposition
   - Eigenvalue problems and geometric interpretations

## Development Notes

- Content is written in Japanese (educational material for Japanese statistics exam)
- Heavy use of mathematical notation in markdown cells
- Statistical visualizations use japanize-matplotlib for proper Japanese font rendering
- Code includes detailed mathematical explanations alongside implementations

## Dependencies Management

The project maintains both `pyproject.toml` and `requirements.txt`. The uv.lock file indicates uv is the preferred package manager for development.