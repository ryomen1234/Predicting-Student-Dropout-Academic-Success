from src.inference.loader import ArtifactLoader

def test_loader_module():
    # feature_names = ArtifactLoader.get_feature_names
    le = ArtifactLoader.get_encoder()

    # print(feature_names)
    print(le)

    print(le.classes_)
    idx_to_class = {idx:label for idx, label in enumerate(le.classes_)}
    print(idx_to_class)

if __name__ == "__main__":
    test_loader_module()

