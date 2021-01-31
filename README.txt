SICK BERT-base, batch_size=16, train:test=8:2,lr=5e-5,epoch=4,acc:0.8653, 0.8730, 0.8643, 0.8750, 0.8740, not fix test set
SICK BERT-base, batch_size=16, train:test=8:2,lr=5e-5,epoch=4,acc:0.8684, 0.8689, 0.8735, 0.8684, 0.8694, fix test set
SICK BERT-base, batch_size=16, train:test=8:2,lr=5e-5,epoch=5,acc:0.8796
SICK BERT-base, batch_size=16, train:test=8:2,lr=5e-5,epoch=5,w/o local encoder:0.8648, with local encoder:0.8831

SICK BERT-base, batch_size=16, train:test=8:2,lr=5e-5,epoch=5

Model  | Accuracy  |
 ---- | ----- | ------  
 w/o local encoding  | 0.8750 | 0.8755 | 0.8709
 with local encoding  | 0.8816 | 0.8765 | 0.8765

AGNews BERT-base, batch_size=16, lr=5e-5, epoch=5, acc:0.9455, 0.9405

AGNews first 10000 cases, BERT-base, batch_size=16, lr=5e-5, epoch=5, acc:0.9195, 0.9150, 0.9055
AGNews first 10000 cases, BERT-base, batch_size=16, lr=5e-5, epoch=5, w/o local encoding:0.9195, 0.9185, with local encoder:0.9230, 0.9195
AGNews first 10000 cases, BERT-base, batch_size=16, lr=5e-5, epoch=4, with label embedding, acc:0.9205
AGNews first 10000 cases, BERT-base, batch_size=16, lr=5e-5, epoch=5, with label embedding, acc:0.9265

AGNews first 10000 cases, BERT-base, batch_size=16, lr=5e-5, epoch=2, with label embedding, acc:0.9155

Cosine Similarity between 4 label embedding vectors:

     C0 | C1 | C2 | C3 
C0 | 1 | -0.0253 | -0.0510 | 0.0198 
C1 | -0.0253 | 1 | 0.0024 | 0.0160
C2 | -0.0510 | 0.0024 | 1 | -0.0037
C3 | 0.0198 | 0.0160 | -0.0037 | 1
