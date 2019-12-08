from load_local_data import create_torchtext_data_object, load_dataset
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from CNN import CNN
import argparse
from utils import convert_embed_to_vec
from tqdm import tqdm



def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.cuda(device=device)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in tqdm(enumerate(train_iter)):
        text = batch.sentence[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not 32):  # One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects / len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1

        if steps % 100 == 0:
            print(
                f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')

        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss / len(train_iter), total_epoch_acc / len(train_iter)


def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_iter)):
            text = batch.text[0]
            if (text.size()[0] is not 32):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects / len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss / len(val_iter), total_epoch_acc / len(val_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pytorch version of CNN modal sense classification', add_help=False,
                                     conflict_handler='resolve')

    parser.add_argument('--vector_file', default='../embeddings/GoogleNews-vectors-negative300.vec',
                        help='path to text file of w2v. If no such file exists, utils.py can create it from '
                             'a binary file.')

    parser.add_argument('--dataset_path', default="./epos_data", help="dir path where train.csv and test.csv are located."
                                                                      "(created with utils.py)")

    parser.add_argument('--cuda', default='0')

    argv = parser.parse_args()

    learning_rate = 1e-3
    batch_size = 50
    output_size = 2
    hidden_size = 256
    embedding_length = 300


    print("preparing data ...")
    train, valid, test = create_torchtext_data_object(argv.dataset_path)
    try:
        TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_dataset(train_data=train,
                                                                                        val=valid, test=test,
                                                                                        embed_fp=argv.vector_file)
    except FileNotFoundError:
        print('Converting binary w2v file to text file.')
        destination_file = argv.vector_file + ".vec"
        convert_embed_to_vec(argv.vector_file.replace(".vec", ".bin"), destination_file)
        TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_dataset(train_data=train,
                                                                                        val=valid, test=test,
                                                                                        embed_fp=destination_file)

    print("Building model ...")
    model = CNN(batch_size=batch_size, output_size=output_size, in_channels=1, out_channels=1, kernel_heights=[3, 4, 5],
                stride=1, padding=1, keep_probab=0.5, vocab_size=vocab_size, embedding_length=embedding_length,
                weights=word_embeddings)

    device = torch.device(f"cuda:{argv.cuda}") if torch.cuda.is_available() else "cpu"
    print(f'Device is: {device}')
    model.to(device)

    loss_fn = F.cross_entropy

    for epoch in range(10):
        print(f"Epoch {epoch}")
        train_loss, train_acc = train_model(model, train_iter, epoch)
        val_loss, val_acc = eval_model(model, valid_iter)

        print(
            f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

    test_loss, test_acc = eval_model(model, test_iter)
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

    ''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
    test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
    test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."

    test_sen1 = TEXT.preprocess(test_sen1)
    test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]

    test_sen2 = TEXT.preprocess(test_sen2)
    test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]

    test_sen = np.asarray(test_sen1)
    test_sen = torch.LongTensor(test_sen)
    test_tensor = Variable(test_sen, volatile=True)
    test_tensor = test_tensor.cuda()
    model.eval()
    output = model(test_tensor, 1)
    out = F.softmax(output, 1)
    if (torch.argmax(out[0]) == 1):
        print("Sentiment: Positive")
    else:
        print("Sentiment: Negative")
