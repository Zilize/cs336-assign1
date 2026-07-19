# a.

#params in token_embeddings:
vocab_size * d_model = 50257 * 1600 = 80411200

#params in ln_final:
d_model = 1600

#params in lm_head:
vocab_size * d_model = 50257 * 1600 = 80411200

#params in a transformer block:
    #params in 2 * ln:
    2 * d_model = 2 * 1600 = 3200
    #params in attn:
    4 * d_model * d_model = 4 * 1600 * 1600 = 10240000
    #params in ffn:
    3 * d_model * d_ff = 3 * 1600 * 4288 = 20582400
    totally: 30825600

#params totally:
80411200 + 1600 + 80411200 + 48 * 30825600 = 1640452800

bytes totally:
1640452800 * 4 = 6,561,811,200

# b.

d_k = d_v = d_model // num_heads = 64

1. token_embeddings: just indexing, no matmul
2. ln_final: no matmul
3. lm_head: 2 * 1024 * 50257 * 1600 = 164,682,137,600
4. transformer block:
   a. attn:
      i. q/k/v projection: 3 * 2 * 1024 * 1600 * 1600 = 15728640000
      ii. rope: no matmul
      iii. scaled_dot_product_attention:
         25 * (2 * 1024 * 1024 * 64 + 2 * 1024 * 1024 * 64) = 6710886400
      iv. output projection: 2 * 1024 * 1600 * 1600 = 5242880000
      attn totally: 27,682,406,400
   b. ffn:
      i. linear projection: 3 * 2 * 1024 * 1600 * 4288 = 42,152,755,200
   block totally: 69,835,161,600

totally: 3,516,769,894,400

# c.

ffn require the most FLOPs
