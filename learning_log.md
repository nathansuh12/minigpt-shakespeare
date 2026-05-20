# Learning notes

A running log of what I'm learning, things that confused me, and things I want to revisit.

---

- Got the repo running. Bigram baseline trains in less than 10 seconds on CPU. note that input forward function for bigram model feeds in whole input so that we can use feed into our transformer the same way.

- [ ] Why does `view(B*T, C)` work for cross-entropy? by squishing the shape to 2 dimensions, we are able to feed in the input as expected by torch.cross_entropy


- in order to keep gpt predicting next token(one direction prediction), we can use torch.tril, applying a triangular mask and therefore hididng the solutions/next time stamps. We then can divide each row in this mask by torch.sum(a, 1, keepdim=True) essentially normalizing each row and now when the mask is applied, we are essentially taking the mean

weights = torch.tril(torch.ones(T,T)) #time stamps
weights = weights / torch.sum(weights, 1, keepdim=True) #get the averages
weights = weights @ x(where x is the batches by time stamps by chanels or in other words the embeding of each batch size lenght of time t) #gives us masked input

- instead of using this sum function and getting the mean using that, we can instead intialize weights as zeroes and then use torch.ones, set the zeroes to negative infinity and then use softmax in order to give us our weighted mask

trill = torch.tril(torch.ones(T,T))
weights = torch.zeros((T,T))
weights = weight.masked_fill(weights = 0, float('-inf'))
weights = F.softmax(weights, dim=1)

- use residual net to keep the gradients clean by doing x + self attention and x + ffwd. then in order to concatenate but also have the different heads know how to use the information from other heads together, we create another learnable matrix that learns how to combine these together