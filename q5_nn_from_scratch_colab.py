import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

# (a) Load data; encode malignant as 1 and benign as 0.
data = load_breast_cancer()
X = data.data.astype(np.float64)
y = (data.target == 0).astype(np.float64).reshape(-1, 1)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)
mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
sd[sd == 0] = 1.0
X_train = (X_train - mu) / sd
X_val = (X_val - mu) / sd
X_test = (X_test - mu) / sd


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1-y) * np.log(1-p)))

class MLP:
    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2/30), (30, 16)); self.b1 = np.zeros((1,16))
        self.W2 = rng.normal(0, np.sqrt(2/16), (16, 8)); self.b2 = np.zeros((1,8))
        self.W3 = rng.normal(0, np.sqrt(2/8), (8, 1)); self.b3 = np.zeros((1,1))

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, X):
        z1 = X @ self.W1 + self.b1; a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2; a2 = np.maximum(0, z2)
        z3 = a2 @ self.W3 + self.b3; p = sigmoid(z3)
        cache = (X, z1, a1, z2, a2, p)
        return p, cache

    def loss(self, X, y, l2=0.0):
        p, _ = self.forward(X)
        penalty = 0.5*l2*sum(np.sum(W*W) for W in (self.W1,self.W2,self.W3))
        return bce(y, p) + penalty

    def backward(self, y, cache, l2=0.0):
        X, z1, a1, z2, a2, p = cache
        n = X.shape[0]
        dz3 = (p-y)/n
        dW3 = a2.T @ dz3 + l2*self.W3; db3 = dz3.sum(axis=0, keepdims=True)
        da2 = dz3 @ self.W3.T; dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2 + l2*self.W2; db2 = dz2.sum(axis=0, keepdims=True)
        da1 = dz2 @ self.W2.T; dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1 + l2*self.W1; db1 = dz1.sum(axis=0, keepdims=True)
        return [dW1, db1, dW2, db2, dW3, db3]


def grad_check():
    model = MLP(SEED)
    Xs, ys = X_train[:5], y_train[:5]
    p, cache = model.forward(Xs)
    ga = model.backward(ys, cache)
    # Flatten analytic gradients in the same order as parameters.
    ga_flat = np.concatenate([g.ravel() for g in ga])
    eps = 1e-5
    gn_flat = []
    for param in model.params():
        for idx in np.ndindex(param.shape):
            old = param[idx]
            param[idx] = old + eps; lp = model.loss(Xs, ys)
            param[idx] = old - eps; lm = model.loss(Xs, ys)
            param[idx] = old
            gn_flat.append((lp-lm)/(2*eps))
    gn_flat = np.asarray(gn_flat)
    return np.linalg.norm(ga_flat-gn_flat)/(np.linalg.norm(ga_flat)+np.linalg.norm(gn_flat)+1e-30)


def train(model, optimizer='sgd', lr=0.01, epochs=150, batch_size=32,
          l2=0.0, patience=None, restore_best=False, seed=42):
    rng = np.random.default_rng(seed)
    params = model.params()
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    velocity = [np.zeros_like(p) for p in params]
    tr_hist, va_hist = [], []
    best_loss = float('inf'); best_params = None; best_epoch = 0; wait = 0
    for epoch in range(1, epochs+1):
        perm = rng.permutation(len(X_train))
        for start in range(0, len(perm), batch_size):
            ids = perm[start:start+batch_size]
            _, cache = model.forward(X_train[ids])
            grads = model.backward(y_train[ids], cache, l2=l2)
            if optimizer == 'sgd':
                for p, g in zip(params, grads): p -= lr*g
            elif optimizer == 'momentum':
                for i, (p,g) in enumerate(zip(params,grads)):
                    velocity[i] = 0.9*velocity[i] + g
                    p -= lr*velocity[i]
            elif optimizer == 'rmsprop':
                for i,(p,g) in enumerate(zip(params,grads)):
                    v[i] = 0.9*v[i] + 0.1*g*g
                    p -= lr*g/(np.sqrt(v[i])+1e-8)
            elif optimizer == 'adam':
                t = (epoch-1)*int(np.ceil(len(X_train)/batch_size)) + start//batch_size + 1
                for i,(p,g) in enumerate(zip(params,grads)):
                    m[i] = 0.9*m[i] + 0.1*g
                    v[i] = 0.999*v[i] + 0.001*g*g
                    mh = m[i]/(1-0.9**t); vh = v[i]/(1-0.999**t)
                    p -= lr*mh/(np.sqrt(vh)+1e-8)
            else: raise ValueError(optimizer)
        tr_hist.append(model.loss(X_train, y_train, l2=l2))
        val_loss = model.loss(X_val, y_val)  # validation data loss, no penalty
        va_hist.append(val_loss)
        if val_loss < best_loss - 1e-10:
            best_loss = val_loss; best_epoch = epoch; wait = 0
            best_params = [p.copy() for p in params]
        else:
            wait += 1
        if patience is not None and wait >= patience: break
    if restore_best and best_params is not None:
        for p, bp in zip(params, best_params): p[...] = bp
    return {'train':tr_hist, 'val':va_hist, 'best_epoch':best_epoch, 'epochs':len(tr_hist),
            'final_train':model.loss(X_train,y_train,l2=l2), 'final_val':model.loss(X_val,y_val),
            'norm':float(np.sqrt(sum(np.sum(W*W) for W in (model.W1,model.W2,model.W3))))}

# (c) gradient check
relerr = grad_check()
print('SPLIT_SIZES', len(X_train), len(X_val), len(X_test))
print('GRADIENT_RELATIVE_ERROR', f'{relerr:.10e}')

# (d) optimizer comparison
opt_results = {}
for opt, lr in [('sgd',0.01),('momentum',0.01),('rmsprop',0.001),('adam',0.001)]:
    model = MLP(SEED)
    res = train(model, optimizer=opt, lr=lr, epochs=150, seed=SEED)
    opt_results[opt] = res
    print('OPT', opt, 'train', f"{res['final_train']:.6f}", 'val', f"{res['final_val']:.6f}", 'best_epoch',res['best_epoch'], 'norm', f"{res['norm']:.4f}")
plt.figure(figsize=(9,6))
for opt, res in opt_results.items():
    e = np.arange(1,len(res['train'])+1)
    plt.plot(e, np.maximum(res['train'],1e-12), label=f'{opt} train')
    plt.plot(e, np.maximum(res['val'],1e-12), linestyle='--', label=f'{opt} validation')
plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Binary cross-entropy loss (log scale)')
plt.title('Optimizer comparison'); plt.legend(ncol=2, fontsize=8); plt.tight_layout(); plt.savefig('q5_optimizers.png',dpi=160); plt.close()

# (e) learning rate experiments
lr_results = {}
for lr in [1e-4,1e-3,1e-2,1e-1,1.0]:
    model=MLP(SEED)
    res=train(model,optimizer='sgd',lr=lr,epochs=150,seed=SEED)
    lr_results[lr]=res
    print('LR',lr,'train',f"{res['final_train']:.6f}",'val',f"{res['final_val']:.6f}")
plt.figure(figsize=(8,5))
for lr,res in lr_results.items():
    plt.plot(np.arange(1,len(res['train'])+1),np.maximum(res['train'],1e-12),label=f'η={lr:g}')
plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Training BCE (log scale)'); plt.title('Plain SGD: learning-rate comparison')
plt.legend(); plt.tight_layout(); plt.savefig('q5_learning_rates.png',dpi=160); plt.close()

# (f) regularization experiments with Adam
reg_results={}
for label,l2,patience,restore in [('none',0.0,None,False),('L2',0.01,None,False),('L2+early stopping',0.01,20,True)]:
    model=MLP(SEED)
    res=train(model,optimizer='adam',lr=0.001,epochs=400,l2=l2,patience=patience,restore_best=restore,seed=SEED)
    reg_results[label]=res
    print('REG',label,'epochs',res['epochs'],'train',f"{res['final_train']:.6f}",'val',f"{res['final_val']:.6f}",'Wnorm',f"{res['norm']:.4f}",'best_epoch',res['best_epoch'])

# (g) Choose L2 + early stopping model, evaluate test once.
final_model=MLP(SEED)
final_res=train(final_model,optimizer='adam',lr=0.001,epochs=400,l2=0.01,patience=20,restore_best=True,seed=SEED)
prob,_=final_model.forward(X_test)
pred=(prob>=0.5).astype(int)
tp=int(np.sum((pred==1)&(y_test==1))); tn=int(np.sum((pred==0)&(y_test==0)))
fp=int(np.sum((pred==1)&(y_test==0))); fn=int(np.sum((pred==0)&(y_test==1)))
accuracy=(tp+tn)/(tp+tn+fp+fn)
precision=tp/(tp+fp) if tp+fp else 0
recall=tp/(tp+fn) if tp+fn else 0
f1=2*precision*recall/(precision+recall) if precision+recall else 0
print('TEST', 'accuracy',f'{accuracy:.4f}','precision',f'{precision:.4f}','recall',f'{recall:.4f}','f1',f'{f1:.4f}')
print('CONFUSION [[TN,FP],[FN,TP]]',[[tn,fp],[fn,tp]])
