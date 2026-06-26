from utils.load_config import load_config

config = load_config()

print(f"config type: {type(config)}")
print(config)

print("access config values")
print(config.data_paths.data_dir_path)
print(config.data_split.test_size)