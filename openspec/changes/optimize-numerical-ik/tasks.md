## 1. Implementation
- [x] 1.1 Replace finite difference Jacobian with `torch.autograd.grad`
- [x] 1.2 Vectorize LMA linear system solver (`torch.linalg.solve` with batch)
- [x] 1.3 Update convergence criteria and default parameters
- [x] 1.4 Verify accuracy with `test_ik_jacobian_method_panda`
