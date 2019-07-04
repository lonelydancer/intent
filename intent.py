import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

train_filename = '../data/a'
#train_filename = '../data/train.data'
#train_filename = '../data/intent.data'
valid_filename = '../data/valid.data'
batch_size = 128
#batch_size = 16
input_dim = 128
hidden_dim = 64
train_model = sys.argv[1]
save_path = '../model/model.test'
epoch_num = 200

cuda_gpu = torch.cuda.is_available()
seed = 0
if cuda_gpu:
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

class IntentModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, voc_size, output_size):
        super(IntentModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.voc_size = voc_size
        self.output_size = output_size
        self.word_embeddings = nn.Embedding(voc_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_size)
#self.softmax = nn.Softmax(dim=1)

    def forward(self, input_sentence):
#        print 'input_sentence', input_sentence
        emb_input = self.word_embeddings(input_sentence)
        if cuda_gpu:
            emb_input = emb_input.cuda()
        batch_size, seq_length, emb_dim = emb_input.shape
#print 'emb', emb_input.shape
        h_0 = torch.zeros(1, batch_size, self.hidden_dim)
        c_0 = torch.zeros(1, batch_size, self.hidden_dim)
        if cuda_gpu:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        output, (final_hidden_state, final_cell_state) = self.lstm(emb_input)
#        print 'shape', output[:,-1,:].shape
        cls = self.linear(output[:,-1,:])#.view(batch_size,-1))
#print 'cls', cls
#        cls = self.softmax(cls)
        return cls

def generate_id(filename):
    voc2id = {'PAD':0}
    voc_list = ['PAD']
    label2id = {}
    label_num = 0
    voc_num =  1
    labels = []
    f = open(filename)
    for line in f:
        data = line.strip('\n').decode('utf8').split('\t')
        if len(data) != 2:
            sys.stderr.write('wrong format [%s]'%(line))
            continue
        label, sent = data
        if label not in label2id:
            label2id[label] = label_num
            label_num += 1
            labels.append(label)
        for v in sent:
            if v not in voc2id:
                voc2id[v] = voc_num
                voc_list.append(v)
                voc_num += 1
    f.close()
    return voc2id, label2id, labels, voc_list


class IntentDataset(Dataset):

    def __init__(self, filename, voc2id, label2id):
        self.voc2id = voc2id
        self.label2id = label2id
        self.x = []
        self.y = []
        f = open(filename)
        self.max_seq_length = 0
        for line in f:
            data = line.strip('\n').decode('utf8').split('\t')
            if len(data) != 2:
                sys.stderr.write('wrong format [%s]\n'%(line))
                continue
            label, sent = data
            if label not in self.label2id:
                sys.stderr.write('oov label:%s\n'%(label))
                continue
            label_id = self.label2id[label]
            voc_id_list = []
            for v in sent:
                if v not in self.voc2id:
                    sys.stderr.write('oov voc:%s\n'%(v.encode('utf8')))
                    continue
                voc_id = self.voc2id[v]
                voc_id_list.append(voc_id)
            self.x.append(torch.tensor(voc_id_list))
            if len(voc_id_list) > self.max_seq_length:
                self.max_seq_length = len(voc_id_list)
            self.y.append(torch.tensor(label_id))
        f.close()
        self.x = pad_sequence(self.x, True)
        print '====='
        print 'max_seq_length', self.max_seq_length
        print 'label_num', len(self.label2id)
        print 'voc_num', len(self.voc2id)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

def eval_precision(data_loader):
    total = 0.0
    correct = 0.0
    model.eval()
    for test_data in data_loader:
        input, label = test_data
        if cuda_gpu:
            input = input.cuda()
            label = label.cuda()
        output = model(input)#.squeeze()
        predict = torch.argmax(output,1)
#print input[:2]
        print output[:2]
        print predict[:2]
        for i in range(len(predict)):
            s = ''
            for k in input[i]:
                if k != 0:
                    s += voc_list[k]
        batch_correct = (label==predict).sum().item()
        batch_total = label.size(0) + 0.0
        total += batch_total
        correct += batch_correct
    return (correct, total, correct/total)

voc2id, label2id, labels, voc_list = generate_id(train_filename)
train_dataset = IntentDataset(train_filename, voc2id, label2id)
train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
#train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

voc_num = len(voc2id)
label_num = len(label2id)
print 'voc_num', len(voc2id)
print 'label_num', len(label2id)

model = IntentModel(input_dim, hidden_dim, voc_num, label_num)



if train_model in ['train','train-continue']:

    best_acc = 0.0

    if cuda_gpu:
        model = model.cuda()

    opt = optim.SGD(model.parameters(), lr=0.1)
#    opt = optim.Adam(model.parameters())

    if train_model == 'train-continue':
        checkpoint = torch.load(save_path+'.best.pt') 
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        best_acc = checkpoint['best_acc']
        print 'best_acc', best_acc
        

    validation_dataset = IntentDataset(valid_filename, voc2id, label2id)
    validation_loader = DataLoader(validation_dataset, batch_size, shuffle=False)
#criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epoch_num):
        total_loss = 0.0
        model.train()
        for idx, (input, label) in enumerate(train_loader):
            opt.zero_grad()
            if cuda_gpu:
                input = input.cuda()
                label = label.cuda()
            output = model(input)#.squeeze()
#print idx,input.shape[0]
            loss = criterion(output, label)
            loss.backward()
            opt.step()
#jfor param in model.parameters():
#              print param.data
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            correct, total, precision = eval_precision(train_loader)
            print 'epoch %d total_loss = %lf' %(epoch, total_loss)
            print 'train data precision:%d/%d = %.2lf%%'% (correct, total, correct/total*100)
            correct, total, test_precision = eval_precision(validation_loader)
            print 'valid data precision:%d/%d = %.2lf%%'% (correct, total, correct/total*100)
            if precision > best_acc:
                torch.save({'epoch':epoch, \
                    'model_state_dict': model.state_dict(),\
                    'optimizer_state_dict':opt.state_dict(),
                    'loss': total_loss,
                    'best_acc': precision},\
                            save_path+'.best.pt')
                best_acc = precision
                print 'save model'
else:
    cuda_gpu = False
    opt = optim.SGD(model.parameters(), lr=0.9)
    checkpoint = torch.load(save_path+'.best.pt') 
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    print 'train predict'
    correct, total, precision = eval_precision(train_loader)
    print correct, total, precision
    while True:
        line = sys.stdin.readline()
        if line == "":
            break
        data  = line.strip('\n').decode('utf8').split()
        query = data[1]
        label = data[0]
        input = []
        for i in query:
            if i in train_dataset.voc2id:
                input.append(train_dataset.voc2id[i])
        input = [input+[0]*(train_dataset.max_seq_length-len(input))]
        h = model(torch.tensor(input)).squeeze()
        idx = torch.argmax(h,0).item()
        print  str(label.encode('utf8') == labels[idx].encode('utf8')) + '\t' + line.strip('\n') + '\t' + str(labels[idx]) + '\t' + str(torch.max(h).item())
