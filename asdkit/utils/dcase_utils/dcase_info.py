def get_dcase_info(path: str, label: str) -> str | int:
    # section_00_target_train_normal_0009_<attribute>.wav
    split_path = path.split("/")[-1].split("_")
    if label == "split":
        return path.split("/")[-2]
    elif label == "machine":
        return path.split("/")[-3]
    elif label == "section":
        return int(split_path[1])
    elif label == "is_target":
        if len(split_path) < 5:
            return -1
        assert split_path[2] in ["source", "target"]
        return int(split_path[2] == "target")
    elif label == "is_normal":
        if len(split_path) < 5:
            return -1
        assert split_path[4] in ["normal", "anomaly"]
        return int(split_path[4] == "normal")
    elif label == "attr":
        if len(split_path) < 7:
            return ""
        return "_".join(split_path[6:]).replace(".wav", "")
    else:
        raise ValueError(f"Unknown label: {label}")
