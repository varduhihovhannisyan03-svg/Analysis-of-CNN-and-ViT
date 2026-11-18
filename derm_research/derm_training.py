"""Shared training functions for DERM12345 notebooks. PyTorch, timm, CPU/GPU compatible."""
import json, time, random, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, roc_auc_score, confusion_matrix,
    classification_report)

LABELS=['bdb','cb','db','srjd']
CLASS_NAMES=['Blue nevus','Compound nevus','Dermal nevus','Spitz/Reed nevus']
SEED=42

def seed_all(seed=SEED):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def resolve_data(results_dir=Path('results')):
    path=Path(results_dir)/'selected_four_classes.csv'
    if not path.exists():raise FileNotFoundError(f'{path} not found. Run 01_Preprocessing first.')
    df=pd.read_csv(path)
    needed={'partition','label','path','patient_id','image_id'}
    if not needed.issubset(df):raise ValueError(f'Missing columns: {needed-set(df.columns)}')
    df=df[df.label.isin(LABELS)].copy()
    if set(df.partition.unique())!={'train','val','test'}:raise ValueError('Expected train, val and test partitions')
    if df.image_id.duplicated().any():raise ValueError('Duplicated image IDs')
    sets={p:set(df.loc[df.partition==p,'patient_id'].dropna()) for p in ('train','val','test')}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if sets[a]&sets[b]:raise ValueError(f'Patient leakage: {a}/{b}')
    missing=[p for p in df.path if not Path(p).exists()]
    if missing:raise FileNotFoundError(f'{len(missing)} images missing. First path: {missing[0]}. Check Jupyter working directory.')
    return df

class DermDataset(Dataset):
    def __init__(self,frame,transform):self.frame=frame.reset_index(drop=True);self.transform=transform
    def __len__(self):return len(self.frame)
    def __getitem__(self,index):
        row=self.frame.iloc[index]
        with Image.open(row.path) as im:x=self.transform(im.convert('RGB'))
        return x,LABELS.index(row.label)

def loaders_for(df,batch_size=8,image_size=224,num_workers=0):
    norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    train_tf=transforms.Compose([transforms.Resize((image_size,image_size)),
        transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=.08,contrast=.08,saturation=.05),
        transforms.ToTensor(),norm])
    test_tf=transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor(),norm])
    return {p:DataLoader(DermDataset(df[df.partition==p],train_tf if p=='train' else test_tf),
         batch_size=batch_size,shuffle=(p=='train'),num_workers=num_workers)
         for p in ('train','val','test')}

def evaluate(model,loader,device,criterion):
    model.eval();loss_sum=0.;ys=[];probs=[]
    with torch.inference_mode():
        for x,y in loader:
            logits=model(x.to(device));loss=criterion(logits,y.to(device))
            loss_sum+=float(loss.item())*len(y)
            ys.extend(y.numpy().tolist());probs.append(torch.softmax(logits,1).cpu().numpy())
    ys=np.array(ys);probs=np.concatenate(probs);pred=probs.argmax(1)
    score=precision_recall_fscore_support(ys,pred,labels=[0,1,2,3],zero_division=0)[2].mean()
    return {'loss':loss_sum/len(ys),'accuracy':accuracy_score(ys,pred),'macro_f1':float(score)},ys,probs

def train_one(name, *, head_epochs=3, finetune_epochs=8, batch_size=8,
              lr_head=1e-3,lr_finetune=2e-5,patience=4,image_size=224,
              results_root='results_v2', num_workers=0):
    seed_all()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df=resolve_data();loaders=loaders_for(df,batch_size,image_size,num_workers)
    root=Path(results_root)/name;root.mkdir(parents=True,exist_ok=True)
    model=timm.create_model(name,pretrained=True,num_classes=4).to(device)
    counts=df.loc[df.partition=='train','label'].value_counts().reindex(LABELS).to_numpy()
    w=1/np.sqrt(counts.astype(float));w=w/w.mean()
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=device))
    # Stage 1: freeze feature extractor, train only classification head.
    for param in model.parameters():param.requires_grad=False
    classifier=model.get_classifier()
    for param in classifier.parameters():param.requires_grad=True
    if not any(p.requires_grad for p in model.parameters()):raise RuntimeError('Classifier was not unfrozen')
    history=[];best_score=-1.;bad=0;best_epoch=None
    checkpoint=root/'best.pt';start=time.perf_counter();global_epoch=0
    def run_stage(phase,epochs,lr):
        nonlocal best_score,bad,best_epoch,global_epoch
        optimizer=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=lr,weight_decay=1e-4)
        for ep in range(epochs):
            global_epoch+=1
            # Frozen backbone should retain eval state (especially BatchNorm)
            model.train()
            if phase=='head':
                for module in model.modules():
                    if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
            total=0.;correct=0;n=0
            for x,y in loaders['train']:
                x=x.to(device);y=y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits=model(x);loss=criterion(logits,y);loss.backward();optimizer.step()
                total+=float(loss.item())*len(y);correct+=int((logits.argmax(1)==y).sum().item());n+=len(y)
            val,_,_=evaluate(model,loaders['val'],device,criterion)
            row=dict(epoch=global_epoch,phase=phase,train_loss=total/n,train_accuracy=correct/n,
                     val_loss=val['loss'],val_accuracy=val['accuracy'],val_macro_f1=val['macro_f1'])
            history.append(row)
            pd.DataFrame(history).to_csv(root/'history.csv',index=False)
            print(f'{name} | {phase} {ep+1}/{epochs} | train_loss={row["train_loss"]:.4f} '
                  f'val_loss={row["val_loss"]:.4f} val_macro_f1={row["val_macro_f1"]:.3f}',flush=True)
            if val['macro_f1']>best_score+1e-4:
                best_score=val['macro_f1'];best_epoch=global_epoch;bad=0
                torch.save(model.state_dict(),checkpoint)
            else:bad+=1
            if bad>=patience:
                print('Early stopping triggered (validation macro F1).');return True
        return False
    print('Device:',device,'Model:',name,'class weights:',dict(zip(LABELS,w.round(3))))
    run_stage('head',head_epochs,lr_head)
    # Restart patience when beginning finetuning, even if head stage plateaued.
    bad=0
    for p in model.parameters():p.requires_grad=True
    if finetune_epochs>0:run_stage('fine_tune',finetune_epochs,lr_finetune)
    training_seconds=round(time.perf_counter()-start,2)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    # Evaluate test ONCE after hyperparameters and best checkpoint chosen on validation.
    test,y,probs=evaluate(model,loaders['test'],device,criterion);pred=probs.argmax(1)
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=[0,1,2,3],zero_division=0)
    metrics={'model':name,'training_seconds':training_seconds,'best_epoch':best_epoch,
        'epochs_completed':global_epoch,'accuracy':float(test['accuracy']),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'macro_precision':float(precision.mean()),'macro_recall':float(recall.mean()),
        'macro_f1':float(f1.mean())}
    try:metrics['macro_roc_auc_ovr']=float(roc_auc_score(y,probs,multi_class='ovr',average='macro',labels=[0,1,2,3]))
    except ValueError:metrics['macro_roc_auc_ovr']=None
    for k,lbl in enumerate(LABELS):
        metrics[f'{lbl}_precision']=float(precision[k]);metrics[f'{lbl}_recall']=float(recall[k]);metrics[f'{lbl}_f1']=float(f1[k]);metrics[f'{lbl}_support']=int(support[k])
    predictions=df[df.partition=='test'][['image_id','patient_id','label','path']].reset_index(drop=True).copy()
    predictions['true_index']=y;predictions['pred_index']=pred;predictions['pred_label']=[LABELS[i] for i in pred]
    for k,lbl in enumerate(LABELS):predictions[f'prob_{lbl}']=probs[:,k]
    predictions.to_csv(root/'predictions.csv',index=False)
    (root/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (root/'classification_report.txt').write_text(classification_report(y,pred,target_names=CLASS_NAMES,labels=[0,1,2,3],zero_division=0),encoding='utf-8')
    (root/'config.json').write_text(json.dumps({'model':name,'seed':SEED,'image_size':image_size,
        'batch_size':batch_size,'head_epochs':head_epochs,'finetune_epochs':finetune_epochs,
        'lr_head':lr_head,'lr_finetune':lr_finetune,'patience':patience,
        'selection':'validation_macro_f1','class_weight':'inverse_sqrt_frequency',
        'pretrained':True,'device':str(device)},indent=2),encoding='utf-8')
    print('\nTest metrics:',pd.Series(metrics).to_string())
    print('\n', (root/'classification_report.txt').read_text(encoding='utf-8'))
    return metrics
