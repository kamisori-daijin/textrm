from torch.utils.data import Dataset, DataLoader
import torch
from datasets import load_dataset
from models.trm_build import RMSNorm, TransformerBlock, apply_rotary_pos_emb, RotaryEmbedding
from models.trm_model import TinyRecursiveModel
from models.config import config
from dataset.tinystories import TinyStoriesDataset 
from training.instantiate import tokenizer, device, model
from training.trainer import train
from ema.ema import EMA



if __name__ == '__main__':
    train_dataset = TinyStoriesDataset(
        tokenizer,
        split='train',
        max_length=config['max_seq_len'] + 1,  # +1 for next token prediction
        max_samples=config['max_train_samples']
    )
    val_dataset = TinyStoriesDataset(
        tokenizer,
        split='validation',
        max_length=config['max_seq_len'] + 1,
        max_samples=config['max_val_samples']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=2,
        pin_memory=True
    )

    #Training
    save_path = 'best_model.pt' # Path to save the best model
    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        epochs=config['epochs'],
        lr=config['lr'],
        warmup_steps=config['warmup_steps'],
        n_supervision_steps=config['n_supervision_steps'],
        save_path=save_path,
    )
    
    print('\nTraining complete!')
    torch.save(model.state_dict(), 'final_model.pt') # Save final model
    print('Saved final model to final_model.pt')


    # test Generation
    model.eval()
    ema = EMA(model)

    prompts = [
        "Once upon a time",
        "The little girl",
        "One day, a rabbit",
        "Tom and his friend"
    ]

    print('\n=== Generated Stories ===\n')
    for prompt in prompts:
        prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        generated = model.generate(prompt_ids, max_new_tokens=150, temperature=0.8)
        text = tokenizer.decode(generated[0].tolist())
        print(f'Prompt: "{prompt}"')
        print(f'Story: {text}\n')
        print('-' * 50 + '\n')
