def test_shape_dataset():
    from aide.scaffold.dataset import ProceduralShapesDataset

    dataset = ProceduralShapesDataset(num_samples=10, img_size=32, noise_level=0.1)

    assert len(dataset) == 10

    for i in range(len(dataset)):
        img, label = dataset[i]
        assert img.shape == (3, 32, 32)  # CHW format
        assert 0 <= label < 3


def test_shape_dataset_generation(tmp_path):
    from aide.scaffold.dataset import create_procedural_shapes_artifacts

    # Create the procedural shapes dataset artifacts
    manifest_path = create_procedural_shapes_artifacts(str(tmp_path), seed=42)

    # Check that the manifest file exists
    assert manifest_path.is_file()

    # Load the manifest and check its contents
    import json

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "train" in manifest
    assert "val" in manifest
    assert "test" in manifest
    assert "meta" in manifest

    # Check that the dataset files exist
    train_path = tmp_path / "procedural_shapes" / "train.pt"
    val_path = tmp_path / "procedural_shapes" / "val.pt"
    test_path = tmp_path / "procedural_shapes" / "test.pt"

    assert train_path.is_file()
    assert val_path.is_file()
    assert test_path.is_file()
