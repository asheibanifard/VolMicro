from load_model import load_model
model = load_model('pretrained_models/v019/checkpoints/checkpoint_final.pt')
print("Model loaded successfully!")
print(f"  Positions shape: {model.positions_raw.shape}")