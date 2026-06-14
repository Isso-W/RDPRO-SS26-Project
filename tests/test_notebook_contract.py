import ast
import json
from pathlib import Path


def test_integration_notebook_is_colab_entrypoint_without_hyperparameter_overrides():
    path = Path("integration_update_colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["metadata"]["accelerator"] == "GPU"
    text = path.read_text(encoding="utf-8")
    assert "mcp_knowledge" in text
    assert "execute_dog_breed_workflow" in text
    assert "submit_to_kaggle=True" in text
    assert "KAGGLE_API_TOKEN" in text
    for forbidden in (
        "EPOCHS =",
        "BATCH_SIZE =",
        "LEARNING_RATE =",
        "BACKBONE =",
        "IMAGE_SIZE =",
    ):
        assert forbidden not in text
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.lstrip().startswith("%"):
            continue
        ast.parse(source, filename=f"cell-{index}")
