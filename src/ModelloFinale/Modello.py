import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class IronyEndToEndModel(nn.Module):
 
    def __init__(self, 
                 dim_vocabolario,      # dimensione del vocabolario (usiamo uno unico per tutti i dataset ITA + MC)
                 dim_wordVector=128,   # dimensione del wordVector (128 feature)
                 dim_frase=64,         # lunghezza massima della frase (64 parole)
                 num_heads=4):         # numero di teste della MHSA    
        super().__init__()


        # Istanzio i layer senza fare il forward

        # EMBEDDING LAYER 
        self.embedding_layer = EmbeddingLayer(      
            dim_vocabolario=dim_vocabolario,
            dim_wordVector=dim_wordVector,
            dim_frase=dim_frase,
            padding_idx=0
        )

        # MHSA LAYER
        self.mhsa = MultiHeadSelfAttention(      
            dim_model=dim_wordVector,
            num_heads=num_heads
        )

        # PRIMO RESIDUALNORM LAYER
        self.residual1 = ResidualLayerNorm(      
            dim_model=dim_wordVector
        )
        

        # FFN LAYER
        self.ffn = PositionWiseFFN(              
            dim_model=dim_wordVector
        )

        # SECONDO RESIDUALNORM LAYER
        self.residual2 = ResidualLayerNorm(      
            dim_model=dim_wordVector
        )

        # MEAN POOLING LAYER
        self.mean_pool = MeanPooling()            


        # CLASSIFICATORE FINALE
        self.classifier = SentenceClassifier(       
            dim_model=dim_wordVector
        )

    # Chiama in automatico i forward di tutti i layer
    def forward(self, input_ids, mask):

        # !! input_ids è un tensore MATRICE (numero di frasi x 64 parole(ID) per frase)
        # 4000x64 se metto tutto il training set
        # 32x64 se metto solo un subset di 32 frasi

        # Embedding Layer 
        embeddings = self.embedding_layer(input_ids) 
    
        # MHSA Layer
        attn_out = self.mhsa(embeddings, mask=mask)

        # Primo ResidualNorm Layer
        z1 = self.residual1(embeddings, attn_out)   
        
        # FFN Layer
        ffn_out = self.ffn(z1)    

        # Secondo ResidualNorm Layer
        z2 = self.residual2(z1, ffn_out)      

        # Mean Pooling Layer
        sent_repr = self.mean_pool(z2, mask)        
        
        # Classificatore Finale
        logits = self.classifier(sent_repr)         
        logits = logits.squeeze(-1)                 

        return logits    
    

class EmbeddingLayer(nn.Module):
    def __init__(self, dim_vocabolario, dim_wordVector, dim_frase, padding_idx=0):
        super().__init__()

        self.token_embedding = nn.Embedding(
                num_embeddings=dim_vocabolario,   # dim vocabolario
                embedding_dim=dim_wordVector,     # 128
                padding_idx=padding_idx           # id = 0 per il padding
            )     
        # i token <PAD> avranno wordVector = [0.0, 0.0, ..., 0.0] - no informazione
        # però verra sommata l'informazione del posVector anche a loro (standard Transformer)

        self.pos_embedding = nn.Embedding(
                num_embeddings=dim_frase,      # 64
                embedding_dim=dim_wordVector   # 128
        )
      
        self.dim_frase = dim_frase              # 64 
        self.dim_wordVector = dim_wordVector   # 128


    def forward(self, input_ids):
        # nel model è il primo layer, riceve input_ids 
        # cioè una matrice (numero di frasi x 64 parole(ID) per frase)
        # 4000x64 frasi totali oppure 32x64 se subset di 32 frasi

        # Ottengo le dimensioni del subset, e lunghezza frase
        batch_size, seq_len = input_ids.size()  

        # Genera i wordVectors per ogni parola nella frase
        token_emb = self.token_embedding(input_ids)  
        
        # adatto la dimensione del tensore
        positions = torch.arange(seq_len, device=input_ids.device)    
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)  

        # genera i posVectors per ogni posizione nella frase
        pos_emb = self.pos_embedding(positions)
       
        # Ogni parola avrà un vettore (di 128) info parola + info posizione
        # embeddings = token_emb + pos_emb  (questo è il metodo senza normalizzazione)


        # info parola + info posizione con normalizzazione scalare 
        embeddings = (token_emb * math.sqrt(self.dim_wordVector)) + pos_emb


        return embeddings


class MultiHeadSelfAttention(nn.Module):


    def __init__(self, dim_model=128, num_heads=4):
        super().__init__()

        assert dim_model % num_heads == 0, "dim_model deve essere divisibile per num_heads"

        self.dim_model = dim_model                    
        self.num_heads = num_heads                   
        self.head_dim  = dim_model // num_heads   # feature per head : 128/4 = 32

        # pesi da calcolare
        self.W_q = nn.Linear(dim_model, dim_model) # otteniamo Q = "che tipo di altra parola cerco/affine in una frase?"
        self.W_k = nn.Linear(dim_model, dim_model) # otteniamo K = "che tipo di parola è questa?"
        self.W_v = nn.Linear(dim_model, dim_model) # otteniamo V = "che info porto, quanto pesa quella parola sul contesto?"
        
        self.W_o = nn.Linear(dim_model, dim_model) # trasformazione finale che unisce le varie heads
        


    # viene passato x, dopo embedding layer (per ogni frase ogni parola è un wordVector di 128)
    # x è un tesore 4000x64x128 (numero di frasi x lunghezza frase x dim wordVector)
    def forward(self, x, mask=None):
        
        B, L, D = x.size()   # batch size(4000), lunghezza frase(64), dimensione modello(128)

        # GENERO Q,K,V DALLE MATRICI DEI PESI W_q W_k W_v
        Q = self.W_q(x)  # Q = W_q * X + bq  (sempre 4000x64x128)
        K = self.W_k(x)  # K = W_k * X + bk  (sempre 4000x64x128)
        V = self.W_v(x)  # V = W_v * X + bv  (sempre 4000x64x128)

        # QUI FACCIO LO SPLIT NELLE VARIE HEADS
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # CALCOLO L'ATTENZIONE A
        scores = torch.matmul(Q, K.transpose(-2, -1))  # calcolo A = Q * K^T  (64x64)
        scores = scores / math.sqrt(self.head_dim)  # normalizzo
        
        # uso la mask => Annullo l'attenzione delle parole sui token di padding
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded == 0, float('-inf'))

        # la softmax trasforma i -inf in 0, quindi i padding non avranno influenza su altri token
        A = F.softmax(scores, dim=-1) # applico la softmax, ora l'attenzione è distribuita su tutte le feature


        # CALCOLO L'USCITA M
        M = torch.matmul(A, V) # calcolo M = A * V
        M = M.transpose(1, 2).contiguous().view(B, L, D)
        
        out = self.W_o(M)  
        # Out = W_o * M + b_o  (4000x64x128)
        
        return out
    

class ResidualLayerNorm(nn.Module):
    def __init__(self, dim_model=128): 
        super().__init__()

        self.layer_norm = nn.LayerNorm(dim_model)

    # passo X output del embedding layer e out output del layer precedente (MHSA o FFN)
    def forward(self, x, out):

        # sommo feature per feature,
        # unisco le informazioni del nuovo layer con quello precedente (così non perdo info)
        y = x + out
        
        # normalizzazione layer norm
        z = self.layer_norm(y) 
        # ho i paramtri gamma e beta da addestrare
        
        return z # (4000x64x128)


class PositionWiseFFN(nn.Module):
    def __init__(self, dim_model=128, dim_hidden=None):
        super().__init__()
        
        # Versione potenziata, il primo layer espande le feature a 512
        if dim_hidden is None:
            dim_hidden = dim_model * 4   # = 512

        self.linear1 = nn.Linear(dim_model, dim_hidden)
        self.activation = nn.ReLU()     
        self.linear2 = nn.Linear(dim_hidden, dim_model)

    def forward(self, x):
        z = self.linear1(x)      
        h = self.activation(z)
        y = self.linear2(h)      
        return y
    
    # ritorna lo stesso tensore (4000x64x128) ma le parole saranno cambiate



class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()
        

    # viene passato X (4000x64x128) e la mask (4000x64), 
    # deve ritornare un tensore (4000x128), ogni riga è una frase, non una parola
    def forward(self, x, mask):
       
        mask_expanded = mask.unsqueeze(-1)     
        mask_expanded = mask_expanded.to(x.dtype)    

        # moltiplicazione elemento per elemento (non prodotto scalare)
        # e mask_expanded si adatta e diventa 4000x64x128 (ripetendo la maschera su tutta la riga)
        # trasforma in 0 tutti i vettori delle parole padding => non voglio rappresentarle nella media delle feature
        x_masked = x * mask_expanded        
        
        sum_x = x_masked.sum(dim=1)   
        lengths = mask_expanded.sum(dim=1)          

        lengths = torch.clamp(lengths, min=1e-8)
        sent_repr = sum_x / lengths                  
        return sent_repr   


class SentenceClassifier(nn.Module):
    def __init__(self, dim_model=128):  
        super().__init__()
        
        # singolo strato Fully Connected
        self.linear = nn.Linear(dim_model, 1) 

        
    def forward(self, x):
        
        logits = self.linear(x)   # un valore reale per ogni frase (4000x1)
        
        return logits
























