import argparse
import pytorch_lightning as pl
from model import NERModel
from utils import *
from functools import partial
from dataset import CustomDataset, collate_fn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings

warnings.filterwarnings('ignore')


def train(args):
    pl.seed_everything(args.random_seed)
    tag_to_ix, ix_to_tag = load_tags(args.tag_path)
    model = NERModel(
        pretrained_model=args.pretrained_model,
        tag_to_ix=tag_to_ix,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    train_dataset = CustomDataset(data_path=args.train_data_path, tag_to_ix=tag_to_ix)
    dev_dataset = CustomDataset(data_path=args.dev_data_path, tag_to_ix=tag_to_ix)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  collate_fn=partial(collate_fn, tokenizer=tokenizer, max_len=args.max_len),
                                  )
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=args.eval_batch_size,
                                collate_fn=partial(collate_fn, tokenizer=tokenizer, max_len=args.max_len),
                                )

    checkpoint_callback = ModelCheckpoint(monitor='dev_f1_score',
                                          dirpath=args.checkpoint_path,
                                          filename='{epoch}-{valid_loss:.2f}-{dev_f1_score:.2f}',
                                          save_last=True,
                                          save_top_k=5,
                                          mode='max',
                                          )

    trainer = pl.Trainer(
        devices=8,
        accelerator='gpu',
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        precision=32,
        enable_progress_bar=True
    )

    trainer.fit(model, train_dataloader, dev_dataloader)


def evaluate(args):
    pl.seed_everything(args.random_seed)
    tag_to_ix, ix_to_tag = load_tags(args.tag_path)
    model = NERModel.load_from_checkpoint(args.exp_model_path,
                                          pretrained_model=args.pretrained_model,
                                          tag_to_ix=tag_to_ix,
                                          embedding_dim=args.embedding_dim,
                                          hidden_dim=args.hidden_dim,
                                          lr=args.lr)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    test_dataset = CustomDataset(data_path=args.test_data_path, tag_to_ix=tag_to_ix)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=partial(collate_fn, tokenizer=tokenizer, max_len=args.max_len),
                                 )

    trainer = pl.Trainer(
        devices=1,
        accelerator='gpu',
        check_val_every_n_epoch=1,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        precision=32,
        enable_progress_bar=True
    )

    predictions = trainer.predict(model, test_dataloader, return_predictions=True)
    predictions = torch.hstack(predictions).tolist()
    predictions = [
        [ix_to_tag[tag] for tag in seq if tag != -10000]
        for seq in predictions
    ]
    with open(args.predictions_path, 'w') as f:
        for seq in predictions:
            for tag in seq:
                f.write(tag + '\n')
            f.write('\n')


def main(args):
    if args.train:
        train(args)

    if args.eval:
        evaluate(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='GuWen NER'
    )

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train_batch_size', default=8, type=int, required=False)
    parser.add_argument('--eval_batch_size', default=8, type=int, required=True)
    parser.add_argument('--train_data_path', type=str, required=False)
    parser.add_argument('--dev_data_path', type=str, required=False)
    parser.add_argument('--test_data_path', type=str, required=False)
    parser.add_argument('--tag_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=False)
    parser.add_argument('--pretrained_model', type=str, required=True)
    parser.add_argument('--min_epochs', default=1, type=int, required=False)
    parser.add_argument('--max_epochs', default=10, type=int, required=False)
    parser.add_argument('--max_len', default=512, type=int, required=True)
    parser.add_argument('--lr', default=1e-5, type=float, required=True)
    parser.add_argument('--embedding_dim', default=768, type=int, required=True)
    parser.add_argument('--hidden_dim', default=256, type=int, required=True)
    parser.add_argument('--random_seed', default=3407, type=int, required=True)
    parser.add_argument('--exp_model_path', required=False, type=str)
    parser.add_argument('--predictions_path', required=False, type=str)
    main(parser.parse_args())
