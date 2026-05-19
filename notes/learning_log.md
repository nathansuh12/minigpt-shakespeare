# Learning notes

A running log of what I'm learning, things that confused me, and things I want to revisit.

---

## Day 1 — Setup

- Got the repo running. Bigram baseline trains in ~X minutes on CPU.
- Initial loss is around `-ln(1/vocab_size)` ≈ 4.17 (random guessing over ~65 chars).
- The model "works" but generates gibberish — it has no context beyond the previous character.

## Things I want to understand better

- [ ] Why does `view(B*T, C)` work for cross-entropy?
- [ ] What does `torch.multinomial` actually do?
- [ ] How does the position embedding interact with the token embedding?

## Questions for later

- 
