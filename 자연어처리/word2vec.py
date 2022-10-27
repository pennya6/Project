import sys
sys.path.append("..")
import numpy as np
from commmon.layers import MatMul,SoftWithLoss

class SimpleCBOW:
    def __init__(self,vocab_size,hidden_size):
        V,H=vocab_size,hidden_size
        
        #가중치초기화
        W_in=0.01*np.random.randn(V,H).astype('f')
        W_out=0.01*np.random.randn(H,V).astype('f')
        
        #계층생성
        self.in_layer0=MatMul(W_in)
        self.in_layer1=MatMul(W_in)
        self.out_layer=MatMul(W_out)
        self.Loss_layer=SoftWithLoss()
        
        #모든 가중치와 기울기를 리스트에 모은다.
        layers=[self.in_layer0,self.in_layer1,self.out_layer]
        self.params,self.grads=[],[]
        for layer in layers:
            self.params+=layer.params
            self.grads+=layers.grads
        self.word_vecs=W_in
        
    #순전파
    def forward(self,contexts,target):
        h0=self.in_layer0.forward(contexts[:,0])
        h1=self.in_layer1.forward(contexts[:,1])
        h=(h0+h1)*0.5
        score=self.out_layer.forward(h)
        loss=self.Loss_layer.forward(score,target)
        return loss