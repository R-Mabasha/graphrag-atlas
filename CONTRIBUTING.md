# Contributing to `graphrag-atlas`

Thank you for your interest in contributing to `graphrag-atlas`! We welcome contributions from developers, researchers, and data engineers worldwide.

---

## 🛠️ Development Setup

We use [`uv`](https://github.com/astral-sh/uv) for blazing-fast, deterministic dependency management and packaging.

### 1. Clone the repository
```bash
git clone https://github.com/R-Mabasha/graphrag-atlas.git
cd graphrag-atlas
```

### 2. Install dependencies & virtual environment
```bash
uv sync --all-extras
```

### 3. Run Linter & Formatter
We use [`ruff`](https://github.com/astral-sh/ruff) for linting and code formatting:
```bash
# Check code style and rules
uv run ruff check graphrag_atlas tests

# Autoformat code
uv run ruff format graphrag_atlas tests
```

### 4. Run Test Suite
```bash
uv run pytest
```

---

## 💡 How to Contribute

- **Report Bugs:** Open an issue with reproduction code, system environment, and observed vs expected behavior.
- **Propose Features:** Start a discussion or open an issue outlining the architectural motivation and design.
- **Submit Pull Requests:**
  1. Fork the repository and create your feature branch: `git checkout -b feat/my-new-feature`.
  2. Write clean, modular Python code following project patterns and type annotations.
  3. Include unit tests covering your changes in `tests/`.
  4. Ensure all linters and tests pass: `uv run ruff check graphrag_atlas tests && uv run pytest`.
  5. Commit with conventional commit messages (e.g., `feat(retrieval): ...`, `fix(lifecycle): ...`).
  6. Push to your fork and submit a PR!

---

## 📜 Code of Conduct

Be welcoming, respectful, and constructive. We strive to maintain a supportive and productive open-source community.

---

## 📄 License

By contributing to `graphrag-atlas`, you agree that your contributions will be licensed under its [MIT License](LICENSE).
