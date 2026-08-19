from aide.scaffold.templates.plugins_data_dataset import ProceduralShapesDataset


def test_shape_dataset():

    dataset = ProceduralShapesDataset(num_samples=10, img_size=32, noise_level=0.1)

    assert len(dataset) == 10

    for i in range(len(dataset)):
        img, label = dataset[i]
        assert img.shape == (3, 32, 32)  # CHW format
        assert 0 <= label < 3
