if __name__ == '__main__':
    import sys
    import torch
    from src.experiments.experiment_configs import experiment_configs
    from torch.utils.data import DataLoader
    from src.modules.utils import train_model
    import torch.backends.cudnn as cudnn

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True
        torch.cuda.empty_cache()

    # Get experiment configurations
    experiment_key = sys.argv[1]
    config = experiment_configs[experiment_key]

    # Set up data and model
    training_dataset = config['dataset'](validation=False, transform=config['transform'])
    training_data_loader = DataLoader(training_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    model = config['model'](**config['model_params'])
    
    # Set up validation dataset
    validation_dataset = config['dataset'](validation=True, transform=config['transform'])
    validation_data_loader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=True,num_workers=4, pin_memory=True)

    optimizer = config['optimizer'](model.parameters(), lr=config['learning_rate'], **config['optimizer_params'])
    scheduler = None
    if config['scheduler']:
        scheduler = config['scheduler'](optimizer, **config['scheduler_params'])

    # Train model using the selected configurations
    train_model(model=model,
                training_data_loader=training_data_loader,
                validation_data_loader=validation_data_loader,
                loss=config['loss'](),
                optimizer=optimizer,
                learning_rate=config['learning_rate'],
                epochs=config['epochs'],
                scheduler=config['scheduler'],
                device=device,
                use_checkpoint=True,
                checkpoint_folder='checkpoints',
                eval_function=config['eval_function'])