# build dataset according to given 'dataset_file'
def build_dataset(args):
    if args.dataset_file == "SHHA":
        from crowd_datasets.SHHA.loading_data import loading_data

        return loading_data

    elif args.dataset_file == "WORM":
        from crowd_datasets.WORM.loading_data import loading_data

        return loading_data

    elif args.dataset_file == "WORM_VAL":
        from crowd_datasets.WORM.loading_data import loading_data_val

        return loading_data_val

    elif args.dataset_file == "WORM_EVAL":
        from crowd_datasets.WORM.loading_data import loading_data_eval
        
        return loading_data_eval

    return None
