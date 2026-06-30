from src.inference.loader import ArtifactLoader

def test_loader_module():
    feature_names = ArtifactLoader.get_feature_names()
    le = ArtifactLoader.get_encoder()

    print(feature_names)
    print(le)

if __name__ == "__main__":
    test_loader_module()

